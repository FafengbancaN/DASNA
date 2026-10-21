"""Train the conditional graph VAE and latent diffusion model on training nodes.

Run with ``python -m dasna.augmentation.train --dataset amazon``. By default,
training is followed by standalone anomaly synthesis at 5%, 10%, and 25% of the
number of training anomalies. The old graph-merging export has been retired.

Modified for DASNA: package CLI, safe checkpoint format and standalone export.
"""
from __future__ import annotations

import argparse

from dasna.augmentation.common import (
    DATASET_FILES, DEFAULT_DATASETS, add_common_arguments, load_mat_with_masks,
    make_loader, positive_int, resolve_device, seed_everything, training_metadata,
    train_ae, train_dm,
)
from dasna.augmentation.diffusion import MLPDiffusion, Model
from dasna.augmentation.models import VGAE


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.set_defaults(batch_size=2048)
    parser.add_argument("--vae-epochs", "--vae_epochs", type=positive_int, default=20)
    parser.add_argument("--diff-epochs", "--diff_epochs", type=positive_int, default=20)
    parser.add_argument("--patience", type=positive_int, default=10)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--wx", type=float, default=1.0)
    parser.add_argument("--we", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Skip training and synthesize from existing checkpoints.")
    parser.add_argument("--train-only", action="store_true", help="Train checkpoints without synthesizing output.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lr <= 0 or args.wd < 0:
        parser.error("--lr must be positive and --wd non-negative")
    if min(args.wx, args.we, args.beta) < 0:
        parser.error("--wx, --we, and --beta must be non-negative")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must lie strictly between 0 and 1")
    if args.sample_steps < 2:
        parser.error("--sample-steps must be at least 2")
    if args.resume and args.train_only:
        parser.error("--resume and --train-only cannot be combined")
    seed_everything(args.seed)
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    completed = 0
    for name in (args.dataset,) if args.dataset else DEFAULT_DATASETS:
        path = args.data_dir / DATASET_FILES[name]
        if not path.is_file():
            if args.dataset:
                parser.error(f"Data file not found: {path}")
            print(f"[{name}] SKIP: data file not found: {path}")
            continue
        try:
            data = load_mat_with_masks(path)
            print(f"[{name}] nodes={data.num_nodes}, features={data.num_node_features}, "
                  f"training anomalies={int((data.y[data.train_mask] == 1).sum())}")
            if not args.train_only:
                from dasna.augmentation.generate import check_output_paths, generate_dataset
                check_output_paths(name, args)
            if not args.resume:
                sub, metadata = training_metadata(data, name, args)
                if metadata["n_train_anomalies"] == 0:
                    raise ValueError("Training mask has no anomalies to model.")
                loader = make_loader(sub, args.batch_size)
                ae = VGAE(metadata["in_dim"], metadata["hid_dim"], temporal=False,
                          threshold=args.threshold).to(device)
                ae = train_ae(ae, loader, args, device, args.checkpoint_dir / f"{name}_ae.pt", metadata)
                dm = Model(MLPDiffusion(metadata["hid_dim"], metadata["diff_dim"]), metadata["hid_dim"]).to(device)
                train_dm(dm, ae, loader, args, device, args.checkpoint_dir / f"{name}_dm.pt", metadata)
                print(f"[{name}] checkpoints: {args.checkpoint_dir}")
            if not args.train_only:
                generate_dataset(data, name, args, device)
        except (ValueError, RuntimeError, FileNotFoundError, OSError) as error:
            parser.error(f"{name}: {error}")
        completed += 1
    if not completed:
        parser.error(f"No dataset files found in {args.data_dir}. See data/README.md for input preparation.")
    print(f"Completed {completed} dataset(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
