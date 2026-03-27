"""Basic training loop for GeoSaccade."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from geosaccade.models.geosaccade import GeoSaccade
from geosaccade.losses.multi_task import GeoSaccadeLoss
from geosaccade.utils.metrics import GeoMetrics
from geosaccade.data.dataset import GeoDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoSaccade")
    parser.add_argument("--data-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--metadata", type=str, required=True, help="CSV metadata file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=5, help="Saccade steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = GeoDataset(
        root_dir=args.data_dir,
        metadata_file=args.metadata,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = GeoSaccade(T=args.T).to(device)
    criterion = GeoSaccadeLoss()
    metrics = GeoMetrics()

    # Optimizer (only train non-frozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    print(f"Model parameters: {sum(p.numel() for p in trainable_params):,} trainable")
    print(f"Dataset size: {len(dataset)}")
    print(f"Device: {device}")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        metrics.reset()

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            targets = batch["coords"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            losses = criterion(outputs, targets)
            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += losses["total"].item()
            metrics.update(outputs["pred_coords"], targets)

            if (batch_idx + 1) % args.log_interval == 0:
                print(
                    f"  [{epoch+1}/{args.epochs}] "
                    f"batch {batch_idx+1}/{len(dataloader)} "
                    f"loss={losses['total'].item():.4f} "
                    f"haversine={losses['haversine'].item():.1f}km"
                )

        scheduler.step()

        # Epoch summary
        epoch_metrics = metrics.compute()
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"loss={epoch_loss/len(dataloader):.4f} "
            f"mean={epoch_metrics['mean_km']:.1f}km "
            f"median={epoch_metrics['median_km']:.1f}km "
            f"acc@25km={epoch_metrics.get('acc@25km', 0):.1f}%"
        )

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = save_dir / f"geosaccade_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_metrics,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    train()
