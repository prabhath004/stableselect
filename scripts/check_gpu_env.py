#!/usr/bin/env python3
"""Print GPU and quantization package versions for cloud notebooks."""

from __future__ import annotations

import importlib.metadata as metadata

import torch


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"bitsandbytes: {package_version('bitsandbytes')}")
    print(f"triton: {package_version('triton')}")
    print(f"transformers: {package_version('transformers')}")


if __name__ == "__main__":
    main()

