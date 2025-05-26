#!/usr/bin/env python
import argparse
import os

from singular_defect import load_llm, i_am_norm
from matplotlib import pyplot as plt
import matplotlib
import warnings
import torch
import numpy as np
from torch import nn
from matplotlib.font_manager import FontProperties

warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

font = FontProperties(fname=Path("assets/unifont.otf"), size=18)
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["figure.figsize"] = [10, 4]
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama2_7b", help="LLaMA model")
    parser.add_argument("--sentence", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--access_token", type=str)
    parser.add_argument("--savedir", type=str, default="output")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    parser.add_argument("--revision", default=None, help="Revision of the model")
    parser.add_argument("--annotate", default=[], nargs="+", type=int, help="annotate tokens")
    parser.add_argument("--layer_id", default=None, type=int, help="layer to plot")
    parser.add_argument("--show_number", action="store_true", help="show number")
    args = parser.parse_args()
    return args


ANNOTATE_MAP = {
    "llama2_7b": [1, 2, 3, 5],
    "llama2_7b_chat": [1, 2, 3, 8],
    "llama2_7b_code": [1, 2, 3, 4],
    "llama2_13b": [1],
    "llama2_13b_chat": [1],
    "llama3_8b": [1],
    "llama3_8b_instruct": [1],
    "phi3_medium": [2],
}

LAYER_ID_MAP = {
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


class ZeroAttention(nn.Module):
    def __init__(self, n_return=2):
        super().__init__()
        self.n_return = n_return

    def forward(self, hidden_states=None, *args, **kwargs):
        if self.n_return == 2:
            return torch.zeros_like(hidden_states), None
        elif self.n_return == 3:
            return torch.zeros_like(hidden_states), None, None
        else:
            raise ValueError("Unsupported n_return")


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
    if args.annotate == []:
        args.annotate = [-1, -2, -3, -4, -5]

    if args.layer_id is None:
        args.layer_id = LAYER_ID_MAP[args.model]

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    for layer in layers:
        if "phi" in args.model:
            layer.self_attn = ZeroAttention(n_return=3)
            model.model.config.use_cache = False
        else:
            layer.self_attn = ZeroAttention()
    print(model)

    layer_output = None

    # Define a hook function to capture the output of a specific layer
    def hook_fn(module, input, output):
        global layer_output
        layer_output = output[0]

    # Choose the layer you want to hook
    layer_to_hook = layers[args.layer_id]  # Change the index to the desired layer

    hook = layer_to_hook.register_forward_hook(hook_fn)

    n_token = max(max(tokenizer.all_special_ids) + 1, tokenizer.vocab_size)
    tokens = model.model.embed_tokens.weight.clone()[:n_token]
    n_token = tokens.size(0)
    trained_mask = tokens.norm(dim=-1) > tokens.norm(dim=-1).mean() / 10

    # Run a forward pass to trigger the hook
    input_ids = torch.arange(n_token).unsqueeze(1).to(device)
    with torch.inference_mode():
        model(input_ids)

    # Unregister the hook
    hook.remove()

    # The output feature of the hooked layer is now stored in `layer_output`
    print(layer_output.shape)

    norms_sorted, ids_sorted = layer_output.norm(dim=-1).flatten().detach().cpu().sort()
    norms = layer_output.norm(dim=-1).flatten().detach().cpu()
    norms[~trained_mask] = np.nan
    plt.scatter(np.arange(n_token), norms, s=4, alpha=0.5)
    plt.gca().set_yscale("log")
    plt.ylim(1, 1e4)

    for i in range(1, max(abs(t) for t in args.annotate) + 1):
        color = "red" if i in args.annotate else "darkgray"
        i = -i
        x_offset = 20 if ids_sorted[i] < len(norms) / 2 else -20
        if args.show_number:
            post = f": {norms_sorted[i]:.1f}"
        else:
            post = ""
        plt.annotate(
            rf"{repr(tokenizer.decode(ids_sorted[i]))}{post}",
            (ids_sorted[i], norms_sorted[i]),
            textcoords="offset points",
            xytext=(x_offset, 5),
            ha="right" if ids_sorted[i] > len(norms) / 2 else "left",
            c=color,
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc3",
                color=color,
            ),
            fontproperties=font,
        )

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    for j in range(1, 32):
        print(
            rf"{j} ({ids_sorted[-j]}): {repr(tokenizer.decode(ids_sorted[-j]))}: {norms_sorted[-j]:.1f}, max norm {i_am_norm(ids_sorted[-j], model, tokenizer, device).max().item():.2f}"
        )

    plt.xlabel("Token ID")
    plt.ylabel(f"Norm after Layer {args.layer_id + 1}")

    plt.tight_layout()

    post = "" if args.revision is None else f"_{args.revision}"
    path = f"{args.savedir}/{args.model}_noninitial{post}.{args.format}"
    print(path)
    plt.savefig(path)

    if args.format == "png":
        os.system(f"convert {path} -trim {path}")
        os.system(f"pngcrush -reduce -brute -ow {path}")
    elif args.format == "pdf":
        os.system(f"pdfcrop {path} {path}")
