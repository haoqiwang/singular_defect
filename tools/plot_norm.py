#!/usr/bin/env python
import argparse
import os

from singular_defect import extract_feats, load_llm
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
    parser.add_argument("--sentence", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--access_token", type=str, default="")
    parser.add_argument("--savedir", type=str, default="output")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    parser.add_argument("--type", type=str, default="3d", choices=["2d", "3d"])
    parser.add_argument("--revision", default=None, help="Revision of the model")
    parser.add_argument("--remove_right_sv", action="store_true")
    parser.add_argument("--token_ids", default=[], nargs="+", type=int, help="plot tokens")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.savedir, exist_ok=True)
    if args.format == "png":
        matplotlib.use("Agg")
    elif args.format == "pdf":
        matplotlib.use("PDF")
    else:
        raise ValueError("Unsupported format")
    if args.type == "2d":
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.spines.top"] = False

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    print(model)

    if args.remove_right_sv:
        right_sv = torch.load(f"{args.savedir}/{args.model}_right_singular.pth", weights_only=True)[args.model][
            "right_singular"
        ].to(device)

        def hook_fn(module, input):
            print(type(input), len(input), input[0].shape)
            input = input[0] - input[0] @ right_sv.to(input[0].dtype)[:, None] @ right_sv.to(input[0].dtype)[None, :]
            return (input,)

        layers[1].post_attention_layernorm.register_forward_pre_hook(hook_fn)

    feats, ids = extract_feats(args.token_ids or args.sentence, model, tokenizer, device, return_ids=True)
    feat_norms = feats.norm(dim=-1).detach().cpu()

    if args.type == "2d":
        for i, m in zip([0, 1, 2, -1], ["o-", ">-", "<-", "D-"]):
            plt.plot(feat_norms[:, i], m, label=tokenizer.decode(ids[:, i]))
            plt.legend(loc="center")
        plt.tight_layout()
    elif args.type == "3d":
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection="3d")
        ax.set_box_aspect((1, 0.55, 0.5))

        tokens = [0, 1, 2, 3, 4, 5, 6, -1][::-1]
        hist = feat_norms[:, tokens].float().detach().cpu().numpy()
        xs = np.arange(hist.shape[0])

        for k, (i, m) in reversed(
            list(enumerate(zip(tokens, ["o-", ">-", "<-", "x-", "D-", "^-", "s-", "v-", "p-"])))
        ):
            ys = hist[:, i]
            ax.plot(xs, ys, m, zs=len(tokens) - k - 1, zdir="y", alpha=0.8)

        ax.set_yticks(
            np.arange(len(tokens)), [tokenizer.decode(ids[:, i]) for i in tokens], rotation=-15, ha="left", va="center"
        )
        ax.set_zticklabels(ax.get_zticklabels(), rotation=-10, ha="left", va="top")

    post = "_no_right_sv" if args.remove_right_sv else ""
    post += "" if args.revision is None else f"_{args.revision}"
    path = f"{args.savedir}/{args.model}_norm_{args.type}{post}.{args.format}"
    print(path)
    plt.savefig(path)

    if args.format == "png":
        os.system(f"convert {path} -trim {path}")
        os.system(f"pngcrush -reduce -brute -ow {path}")
    elif args.format == "pdf":
        os.system(f"pdfcrop {path} {path}")
