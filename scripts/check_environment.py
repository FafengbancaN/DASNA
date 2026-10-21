"""Exercise the runtime needed by one DASNA stage without loading datasets."""

import argparse
from importlib.metadata import PackageNotFoundError, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["augmentation", "detection"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    packages = ["torch", "numpy", "scipy", "scikit-learn"]
    packages += ["torch-geometric", "pygod", "torch-sparse"] if args.stage == "augmentation" else ["PyYAML", "mamba-ssm", "causal-conv1d"]
    for package in packages:
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: not installed")

    import torch
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this environment.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capability: {torch.cuda.get_device_capability(0)}")
    x = torch.ones(4, device=args.device)
    assert (x + x).sum().item() == 8
    print(f"Tensor arithmetic passed on {args.device}.")

    if args.stage == "augmentation":
        from torch_geometric.data import Data
        from torch_geometric.loader import ClusterData
        graph = Data(
            x=torch.ones(4, 2),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        )
        ClusterData(graph, num_parts=2, log=False)
        print("Graph partitioning passed.")
    else:
        if args.device != "cuda":
            raise RuntimeError("Use --device cuda to verify the Mamba CUDA kernels required by DASNA.")
        from mamba_ssm import Mamba
        model = Mamba(d_model=8, d_state=4, d_conv=4, expand=2).to(args.device)
        with torch.no_grad():
            output = model(torch.randn(2, 8, 8, device=args.device))
        assert output.shape == (2, 8, 8) and torch.isfinite(output).all()
        torch.cuda.synchronize()
        print("Mamba forward pass passed on CUDA.")


if __name__ == "__main__":
    main()
