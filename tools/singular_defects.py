#!/usr/bin/env python
import argparse
import os

from singular_defect import layerwise_singular_dir, load_llm
import numpy as np
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama2_7b", help="LLaMA model")
    parser.add_argument("--access_token", type=str)
    parser.add_argument("--savedir", type=str, default="output")
    parser.add_argument("--revision", default=None, help="Revision of the model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.savedir, exist_ok=True)

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    print(model)

    sds = layerwise_singular_dir(args.model, layers)

    post = "" if args.revision is None else f"_{args.revision}"
    path = f"{args.savedir}/{args.model}_sd{post}.pth"
    print(path)

    torch.save({args.model: {"sds": sds}}, path)
