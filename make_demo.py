"""Create small synthetic schema/Parquet fixtures for pipeline verification."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from model import ModelConfig


def demo_schema(without_item_dense=False):
    return {
        "user_int": [[1, 32, 1], [2, 24, 3]],
        "item_int": [[3, 32, 1], [4, 16, 1]],
        "user_dense": [[61, 4], [87, 8]],
        "item_dense": [] if without_item_dense else [[124, 4]],
        "seq": {domain: {"prefix": f"events_{domain[-1]}", "ts_fid": 13,
                         "features": [[11, 32], [12, 16], [13, 0]]}
                for domain in ("seq_a", "seq_b", "seq_c", "seq_d")},
    }


def demo_rows(count=96, schema=None):
    schema = schema or demo_schema()
    rng = np.random.default_rng(20261019)
    rows = []
    for index in range(count):
        timestamp = 1750000000 + index * 60
        row = {"timestamp": timestamp, "label_type": 2 if index % 2 else 0,
               "user_id": f"synthetic-{index:04d}"}
        for group in ("user_int", "item_int"):
            for fid, vocab, width in schema[group]:
                values = rng.integers(1, vocab + 1, width).tolist()
                row[f"{group}_feats_{fid}"] = values[0] if width == 1 else values
        for group in ("user_dense", "item_dense"):
            for fid, width in schema.get(group, []):
                row[f"{group}_feats_{fid}"] = rng.normal(size=width).tolist()
        for domain_idx, (_, spec) in enumerate(schema["seq"].items()):
            length = 0 if index % 8 == 0 else 3 + (index + domain_idx) % 11
            row[f"{spec['prefix']}_11"] = rng.integers(1, 33, length).tolist()
            row[f"{spec['prefix']}_12"] = rng.integers(1, 17, length).tolist()
            row[f"{spec['prefix']}_13"] = (timestamp - 60 * np.arange(1, length + 1)).tolist()
        rows.append(row)
    return rows


def write_demo(output, count=96):
    output = Path(output)
    if (output / "schema.json").exists():
        raise FileExistsError("Demo already exists; choose a fresh directory")
    schema = demo_schema()
    rows = demo_rows(count, schema)
    split = count * 2 // 3
    for name, selected in (("train", rows[:split]), ("valid", rows[split:]),
                           ("infer", [{k: v for k, v in r.items() if k != "label_type"} for r in rows[split:]])):
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(selected), directory / "samples.parquet", row_group_size=16)
    (output / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    ModelConfig(d_model=16, emb_dim=8, num_layers=2, hidden_mult=2, dropout=0,
                user_tokens=2, item_tokens=2, user_queries=2, item_queries=1,
                global_memory=2, domain_memory=1, adapter_rank=4,
                local_window=1, recent_window=2, tail_summaries=2,
                seq_max_lens={d: 12 for d in schema["seq"]},
                recent_budgets={d: 4 for d in schema["seq"]}).save(output / "config.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=96)
    args = parser.parse_args()
    if args.rows < 12:
        raise ValueError("Use at least 12 demo rows")
    print(write_demo(args.output, args.rows))
