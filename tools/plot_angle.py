#!/usr/bin/env python
import argparse
import os

from singular_defect import acute_angles_between
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["figure.figsize"] = [10, 4]
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama2_7b", help="LLaMA model")
    parser.add_argument("--savedir", type=str, default="output")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    parser.add_argument("--revision", default=None, help="Revision of the model")
    parser.add_argument("--annotate", default=[], nargs="+", type=int, help="annotate numbers")
    args = parser.parse_args()
    return args


ANNOTATE_MAP = {
    "phi3_mini": [3, 5, 8, 14, 30],
    "phi3_mini_128k": [3, 5, 8, 14, 30],
    "phi3_medium": [6, 12, 18, 34, 39],
    "phi3.5_mini": [3, 5, 8, 14, 30],
    "llama2_7b": [2, 31],
    "llama2_7b_chat": [2, 31],
    "llama2_7b_code": [2, 31],
    "llama2_13b": [4, 39],
    "llama2_13b_chat": [4, 39],
    "llama3_8b": [2, 32],
    "llama3_8b_instruct": [2, 32],
    "llama3.1_8b_instruct": [2, 32],
    "qwen2_7b": [4, 27],
    "qwen2_7b_instruct": [4, 27],
    "qwen2_7b_math": [4, 27],
    "qwen2_7b_math_instruct": [4, 27],
}

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.savedir, exist_ok=True)
    if args.format == "png":
        matplotlib.use("Agg")
    elif args.format == "pdf":
        matplotlib.use("PDF")
    else:
        raise ValueError("Unsupported format")

    if args.model in ANNOTATE_MAP and args.annotate == []:
        args.annotate = ANNOTATE_MAP[args.model]

    post = "" if args.revision is None else f"_{args.revision}"
    sd_empirical = torch.load(f"{args.savedir}/{args.model}_sd_empirical{post}.pth", weights_only=True)[args.model][
        "sds_empirical"
    ]
    sd = torch.load(f"{args.savedir}/{args.model}_sd{post}.pth", weights_only=True)[args.model]["sds"]
    sds = torch.stack([s["u0"] for s in sd])
    angles = acute_angles_between(sds, sd_empirical)
    plt.scatter(np.arange(len(sds)) + 1, angles, c=angles, cmap="jet")
    plt.ylim(0, 95)
    plt.xlim(0, len(sds) + 1)
    plt.xlabel("Layer")
    plt.ylabel("Angle (degree)")
    plt.tight_layout()

    for l in args.annotate:
        plt.annotate(
            f"{angles[l - 1]:.2f}", (l, angles[l - 1]), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    print(" ".join([f"{i + 1}:{a:.2f}" for i, a in enumerate(angles)]))

    path = f"{args.savedir}/{args.model}_angle{post}.{args.format}"
    print(path)
    plt.savefig(path)

    if args.format == "png":
        os.system(f"convert {path} -trim {path}")
        os.system(f"pngcrush -reduce -brute -ow {path}")
    elif args.format == "pdf":
        os.system(f"pdfcrop {path} {path}")
