#!/usr/bin/env python
import argparse
import os

from singular_defect import layerwise_singular_dir, load_llm
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
    parser.add_argument("--access_token", type=str)
    parser.add_argument("--savedir", type=str, default="output")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    parser.add_argument("--revision", default=None, help="Revision of the model")
    parser.add_argument("--layer_id", default=None, type=int, help="layer to plot")
    args = parser.parse_args()
    return args


LAYER_MAP = {
    "llama2_7b": 1,
    "llama2_7b_chat": 1,
    "llama2_7b_code": 1,
    "llama2_13b": 3,
    "llama2_13b_chat": 3,
    "llama3_8b": 1,
    "llama3_8b_instruct": 1,
    "phi3_mini": 2,
    "phi3_mini_128k": 2,
    "phi3_medium": 5,
    "phi3.5_mini": 2,
    "qwen2_7b": 3,
    "qwen2_7b_instruct": 3,
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

    if args.layer_id is None:
        args.layer_id = LAYER_MAP[args.model]

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    print(model)

    layer = layers[args.layer_id]
    A_mlp = layerwise_singular_dir(args.model, [layer], return_A=True)[0]["A_mlp"]
    F = A_mlp - torch.eye(A_mlp.size(0), device=A_mlp.device)
    u, s, vt = torch.linalg.svd(F)

    def ffn(x):
        with torch.inference_mode():
            return layer.mlp(layer.post_attention_layernorm(x[:, None].half().to(device)))[:, 0]

    out_norms = torch.maximum(ffn(-vt).norm(dim=-1), ffn(vt).norm(dim=-1)).detach().cpu()
    signs = torch.sign(ffn(vt).norm(dim=-1) - ffn(-vt).norm(dim=-1))
    plt.scatter(np.arange(len(out_norms)) + 1, out_norms, s=2)
    plt.annotate(
        f"leading right singular vector",
        (10, out_norms[0] * 0.99),
        textcoords="offset points",
        xytext=(0, -30),
        ha="left",
        c="r",
        arrowprops=dict(
            arrowstyle="-",
            connectionstyle="arc3",
            color="r",
        ),
    )
    plt.xlabel(f"Right Singular Vectors of $F$ in Layer {args.layer_id + 1}")
    plt.ylabel("Output Norm of FFN")

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    post = "" if args.revision is None else f"_{args.revision}"
    path = f"{args.savedir}/{args.model}_right_sv{post}.{args.format}"
    print(path)
    plt.savefig(path)

    if args.format == "png":
        os.system(f"convert {path} -trim {path}")
        os.system(f"pngcrush -reduce -brute -ow {path}")
    elif args.format == "pdf":
        os.system(f"pdfcrop {path} {path}")

    path = f"{args.savedir}/{args.model}_right_sv{post}.pth"
    torch.save({args.model: {"right_sv": vt[0] * signs[0].to(vt.device)}}, path)
