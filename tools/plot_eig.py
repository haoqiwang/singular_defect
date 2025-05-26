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
plt.rcParams["axes.spines.right"] = True
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
    "llama2_7b_chat": [2, 31],
    "llama2_7b_code": [2, 31],
    "llama2_13b": [4, 39],
    "llama2_13b_chat": [4, 39],
    "llama3_8b": [2, 32],
    "llama3_8b_instruct": [2, 32],
    "phi3_mini": [3, 5, 8, 30],
    "phi3_mini_128k": [3, 5, 8, 30],
    "phi3_medium": [6, 12, 18, 34, 39],
    "phi3.5_mini": [3, 5, 8, 30],
    "qwen2_7b": [4, 27],
    "qwen2_7b_instruct": [4, 27],
}

LIM_MAP = {
    "llama2_7b_chat": (-40, 95),
    "llama2_7b_code": (-40, 95),
    "llama2_13b": (-40, 95),
    "llama2_13b_chat": (-40, 95),
    "llama3_8b": (-40, 95),
    "llama3_8b_instruct": (-40, 95),
    "phi3_mini": (-60, 95),
    "phi3_mini_128k": (-60, 95),
    "phi3_medium": (-80, 95),
    "phi3.5_mini": (-60, 95),
}

TICK_MAP = {
    "llama2_7b_chat": [20, 0, -20],
    "llama2_7b_code": [20, 0, -20],
    "llama2_13b": [20, 0, -20],
    "llama2_13b_chat": [20, 0, -20],
    "llama3_8b": [20, 0, -20],
    "llama3_8b_instruct": [20, 0, -20],
    "phi3_mini": [20, 0, -20, -40],
    "phi3_mini_128k": [20, 0, -20, -40],
    "phi3_medium": [20, 0, -20, -40, -60],
    "phi3.5_mini": [20, 0, -20, -40],
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
    if args.model in LIM_MAP:
        lim = LIM_MAP[args.model]
    else:
        lim = (-40, 95)
    if args.model in TICK_MAP:
        ticks = TICK_MAP[args.model]
    else:
        ticks = [20, 0, -20]

    eig_angles = torch.load(f"{args.savedir}/{args.model}_eig.pth", weights_only=True)[args.model]["eig_angles"][:, 0]
    eig_vals = torch.load(f"{args.savedir}/{args.model}_eig.pth", weights_only=True)[args.model]["eig_vals"][:, 0]

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Angle (degree)", color=color)
    ax1.scatter(np.arange(len(eig_angles)) + 1, eig_angles, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(*lim)
    ax1.set_yticks(np.arange(0, 100, 20))

    for l in args.annotate:
        ax1.annotate(
            f"{eig_angles[l - 1]:.1f}",
            (l, eig_angles[l - 1] + 3),
            textcoords="offset points",
            xytext=(0, 20),
            ha="right" if l > len(eig_vals) / 2 else "left",
            c=color,
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc3",
                facecolor="black",
            ),
        )

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = "tab:blue"
    ax2.set_ylabel("Eigenvalue", color=color)  # we already handled the x-label with ax1
    ax2.scatter(np.arange(len(eig_angles)) + 1, eig_vals, color=color, marker="x")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(*lim)
    ax2.set_yticks(ticks)

    for l in args.annotate:
        ax2.annotate(
            f"{eig_vals[l - 1]:.1f}",
            (l, eig_vals[l - 1] - 3),
            textcoords="offset points",
            xytext=(0, -30),
            ha="right" if l > len(eig_vals) / 2 else "left",
            c=color,
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc3",
                facecolor="black",
            ),
        )

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    post = "" if args.revision is None else f"_{args.revision}"
    path = f"{args.savedir}/{args.model}_eig{post}.{args.format}"
    print(path)
    plt.savefig(path)

    if args.format == "png":
        os.system(f"convert {path} -trim {path}")
        os.system(f"pngcrush -reduce -brute -ow {path}")
    elif args.format == "pdf":
        os.system(f"pdfcrop {path} {path}")
