"""Single-process training with explicit validation data or a time boundary."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader

from dataset import ParquetBatches, load_schema, move_to_device, parquet_files
from model import TailScope, ModelConfig


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    labels, probabilities, loss_sum, count = [], [], 0.0, 0
    for batch in loader:
        batch = move_to_device(batch, device)
        with autocast_context(amp):
            logits = model(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite validation logits")
        target = batch["label"]
        loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), target, reduction="sum").item()
        count += target.numel()
        labels.append(target.cpu().numpy())
        probabilities.append(logits.float().sigmoid().cpu().numpy())
    if count == 0:
        raise ValueError("Validation partition is empty; check data paths and timestamp boundary")
    labels, probabilities = np.concatenate(labels), np.concatenate(probabilities)
    auc = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None
    return {"auc": auc, "log_loss": loss_sum / count, "rows": count}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Training Parquet file or directory")
    parser.add_argument("--schema", required=True)
    partition = parser.add_mutually_exclusive_group(required=True)
    partition.add_argument("--valid-data", help="A separate held-out Parquet file or directory")
    partition.add_argument("--split-timestamp", type=int, help="Train < timestamp; validation >= timestamp")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="CUDA bfloat16 autocast")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional smoke-test training step limit")
    parser.add_argument("--query-mode", choices=["dualq", "mean"])
    parser.add_argument("--topology", choices=["semilocal", "full"])
    parser.add_argument("--domain-memory", type=int)
    parser.add_argument("--tail-summaries", type=int)
    parser.add_argument("--tail-pooling", choices=["target", "mean"],
                        help="Candidate-conditioned tail summaries or a matched mean-pooling control")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0 or args.max_steps < 0:
        raise ValueError("Invalid training limits")
    if args.amp and (not args.device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
        raise ValueError("--amp requires a CUDA device with bfloat16 support")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    seed_all(args.seed)
    config, schema = ModelConfig.load(args.config), load_schema(args.schema)
    overrides = {key: getattr(args, key) for key in
                 ("query_mode", "topology", "domain_memory", "tail_summaries", "tail_pooling")
                 if getattr(args, key) is not None}
    config = replace(config, **overrides)
    output = Path(args.output)
    if (output / "best.pt").exists():
        raise FileExistsError("Choose a fresh output directory; best.pt already exists")
    if args.valid_data:
        train_files = {f.resolve() for f in parquet_files(args.data)}
        valid_files = {f.resolve() for f in parquet_files(args.valid_data)}
        if train_files & valid_files:
            raise ValueError("Train and validation files overlap; use a timestamp partition")
    train_data = ParquetBatches(args.data, schema, args.batch_size, config.seq_max_lens,
                                timestamp_max=args.split_timestamp, shuffle=True, seed=args.seed)
    valid_data = ParquetBatches(args.valid_data or args.data, schema, args.batch_size,
                                config.seq_max_lens, timestamp_min=args.split_timestamp)
    train_loader = DataLoader(train_data, batch_size=None, num_workers=args.workers)
    valid_loader = DataLoader(valid_data, batch_size=None, num_workers=args.workers)
    model = TailScope(schema, config).to(args.device)
    # Align training randomness across pooling modes after module initialization.
    seed_all(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    (output / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    run = {"arguments": vars(args), "parameters": sum(p.numel() for p in model.parameters()),
           "torch": str(torch.__version__), "device": args.device,
           "train_partition": "timestamp < cutoff" if args.split_timestamp is not None else "separate files",
           "model_name": "TailScope"}
    (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    best_score, stale, steps = float("-inf"), 0, 0
    for epoch in range(args.epochs):
        train_data.epoch = epoch
        model.train()
        epoch_loss, rows, started = 0.0, 0, time.perf_counter()
        for batch in train_loader:
            batch = move_to_device(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args.amp):
                logits = model(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), batch["label"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            rows += batch["label"].numel()
            epoch_loss += loss.item() * batch["label"].numel()
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        if not rows:
            raise ValueError("Training partition is empty")
        metrics = evaluate(model, valid_loader, args.device, args.amp)
        selection_metric = "auc" if metrics["auc"] is not None else "negative_log_loss"
        score = metrics["auc"] if metrics["auc"] is not None else -metrics["log_loss"]
        record = {"epoch": epoch + 1, "steps": steps, "train_loss": epoch_loss / rows,
                  "train_rows": rows, "validation": metrics,
                  "selection_metric": selection_metric, "seconds": time.perf_counter() - started}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, stale = score, 0
            torch.save({"model": model.state_dict(), "config": asdict(config), "schema": schema,
                        "epoch": epoch + 1, "validation": metrics}, output / "best.pt")
        else:
            stale += 1
        if stale >= args.patience or (args.max_steps and steps >= args.max_steps):
            break


if __name__ == "__main__":
    main()
