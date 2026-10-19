"""Streaming schema-driven Parquet reader with explicit temporal partitions.

Categorical ID convention: positive in-vocabulary IDs are valid;
zero, negative, missing and out-of-vocabulary IDs map to padding. Sequence fields
are reordered together by timestamp and retain the most recent events.
"""
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000, 4320000, 6048000, 7776000,
    11664000, 15552000, 31536000], dtype=np.int64)
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


def load_schema(path):
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in ("user_int", "item_int", "seq"):
        if name not in schema:
            raise ValueError(f"Missing schema group: {name}")
    for name in ("user_int", "item_int", "user_dense", "item_dense"):
        specs = schema.get(name, [])
        if any(spec[-1] < 1 for spec in specs):
            raise ValueError(f"Invalid field width in {name}")
        if len({spec[0] for spec in specs}) != len(specs):
            raise ValueError(f"Duplicate field IDs in {name}")
    for domain, spec in schema["seq"].items():
        if not any(fid != spec.get("ts_fid") for fid, _ in spec["features"]):
            raise ValueError(f"No categorical sequence fields in {domain}")
    return schema


def parquet_files(path):
    path = Path(path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files or any(not f.is_file() for f in files):
        raise FileNotFoundError(f"No Parquet files at {path}")
    return files


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


class ParquetBatches(IterableDataset):
    def __init__(self, path, schema, batch_size=128, seq_max_lens=None,
                 labeled=True, timestamp_min=None, timestamp_max=None,
                 shuffle=False, seed=42):
        super().__init__()
        self.files = parquet_files(path)
        self.schema, self.batch_size = schema, batch_size
        self.seq_max_lens = seq_max_lens or {}
        self.labeled = labeled
        self.timestamp_min, self.timestamp_max = timestamp_min, timestamp_max
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        required = ["timestamp"] + (["label_type"] if labeled else [])
        for group in ("user_int", "item_int", "user_dense", "item_dense"):
            required += [f"{group}_feats_{spec[0]}" for spec in schema.get(group, [])]
        for domain in schema["seq"].values():
            required += [f"{domain['prefix']}_{fid}" for fid, _ in domain["features"]]
        self.required = list(dict.fromkeys(required))
        self.row_groups = []
        for file in self.files:
            parquet = pq.ParquetFile(file)
            missing = set(self.required) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{file.name}: missing columns {sorted(missing)}")
            self.row_groups += [(file, i) for i in range(parquet.metadata.num_row_groups)]

    def _feature_array(self, rows, group):
        specs = self.schema.get(group, [])
        categorical = group.endswith("int")
        output = np.zeros((len(rows), sum(s[-1] for s in specs)),
                          dtype=np.int64 if categorical else np.float32)
        offset = 0
        for spec in specs:
            fid, width = spec[0], spec[-1]
            for index, row in enumerate(rows):
                values = as_list(row.get(f"{group}_feats_{fid}"))[:width]
                output[index, offset:offset + len(values)] = [0 if v is None else v for v in values]
            if categorical:
                segment = output[:, offset:offset + width]
                segment[(segment <= 0) | (segment > spec[1])] = 0
            offset += width
        if not categorical:
            np.nan_to_num(output, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(output)

    def convert(self, rows):
        batch_size = len(rows)
        if any(row["timestamp"] is None for row in rows):
            raise ValueError("Request timestamp is required for temporal splitting")
        timestamp = np.array([row["timestamp"] for row in rows], dtype=np.int64)
        batch = {name: self._feature_array(rows, name)
                 for name in ("user_int", "item_int", "user_dense", "item_dense")}
        batch.update(timestamp=torch.from_numpy(timestamp),
                     hour=torch.from_numpy(((timestamp + 8 * 3600) % 86400 / 3600).astype(np.float32)).unsqueeze(1),
                     user_id=[row.get("user_id") for row in rows], seq={}, seq_mask={}, time_buckets={})
        if self.labeled:
            if any(row["label_type"] is None for row in rows):
                raise ValueError("Labeled data contains a null label_type")
            batch["label"] = torch.tensor([int(row["label_type"] == 2) for row in rows], dtype=torch.float32)
        for domain, spec in self.schema["seq"].items():
            fields = [(fid, vocab) for fid, vocab in spec["features"] if fid != spec.get("ts_fid")]
            maximum = self.seq_max_lens.get(domain, 256)
            ids = np.zeros((batch_size, maximum, len(fields)), dtype=np.int64)
            mask = np.ones((batch_size, maximum), dtype=bool)
            buckets = np.zeros((batch_size, maximum), dtype=np.int64)
            for row_idx, row in enumerate(rows):
                events = [as_list(row.get(f"{spec['prefix']}_{fid}")) for fid, _ in fields]
                size = max(map(len, events), default=0)
                times = as_list(row.get(f"{spec['prefix']}_{spec.get('ts_fid')}"))
                if times and len(times) != size:
                    raise ValueError(f"Misaligned timestamps in {domain}")
                if any(len(values) not in (0, size) for values in events):
                    raise ValueError(f"Misaligned side information in {domain}")
                event_times = np.array([0 if v is None else v for v in times], dtype=np.int64)
                order = np.argsort(-event_times, kind="stable") if times else np.arange(size)
                order = order[:maximum]
                used = len(order)
                mask[row_idx, :used] = False
                for field_idx, ((_, vocab), values) in enumerate(zip(fields, events)):
                    if values:
                        data = np.array([0 if v is None else v for v in values], dtype=np.int64)[order]
                        data[(data <= 0) | (data > vocab)] = 0
                        ids[row_idx, :used, field_idx] = data
                if times:
                    selected_times = event_times[order]
                    ages = np.maximum(timestamp[row_idx] - selected_times, 0)
                    bucket = np.minimum(np.searchsorted(BUCKET_BOUNDARIES, ages) + 1, len(BUCKET_BOUNDARIES))
                    buckets[row_idx, :used] = np.where(selected_times > 0, bucket, 0)
            batch["seq"][domain] = torch.from_numpy(ids)
            batch["seq_mask"][domain] = torch.from_numpy(mask)
            batch["time_buckets"][domain] = torch.from_numpy(buckets)
        return batch

    def __iter__(self):
        groups = list(self.row_groups)
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(groups)
        worker = get_worker_info()
        if worker:
            groups = groups[worker.id::worker.num_workers]
        for file, group in groups:
            parquet = pq.ParquetFile(file)
            columns = self.required + (["user_id"] if "user_id" in parquet.schema_arrow.names else [])
            for record in parquet.iter_batches(batch_size=self.batch_size, row_groups=[group], columns=columns):
                rows = record.to_pylist()
                rows = [r for r in rows if
                        (self.timestamp_min is None or r["timestamp"] >= self.timestamp_min) and
                        (self.timestamp_max is None or r["timestamp"] < self.timestamp_max)]
                if self.shuffle:
                    rng.shuffle(rows)
                if rows:
                    yield self.convert(rows)
