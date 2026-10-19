"""Reload a self-contained checkpoint and write one probability per input row."""
import argparse
import csv
from pathlib import Path

import torch

from dataset import ParquetBatches, move_to_device
from model import TailScope, ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = TailScope(checkpoint["schema"], config).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    data = ParquetBatches(args.data, checkpoint["schema"], args.batch_size,
                          config.seq_max_lens, labeled=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x", encoding="utf-8", newline="") as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(["row_index", "user_id", "timestamp", "probability"])
        for batch in data:
            logits = model(move_to_device(batch, args.device))
            probabilities = logits.float().sigmoid().cpu().tolist()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite predictions")
            for user, timestamp, probability in zip(batch["user_id"], batch["timestamp"].tolist(), probabilities):
                writer.writerow([count, user, timestamp, probability])
                count += 1
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()
