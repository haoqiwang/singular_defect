import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    DistilBertTokenizer,
    DistilBertModel,
    XLNetTokenizer,
    XLNetModel,
    XLMTokenizer,
    XLMModel,
)

from .model_dict import MODEL_DICT_LLMs


def load_llm(args):
    print(f"loading model {args.model}")
    model_name, cache_dir = MODEL_DICT_LLMs[args.model]["model_id"], MODEL_DICT_LLMs[args.model]["cache_dir"]

    if args.model.lower().startswith("bert_"):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).cuda()
        device = "cuda"
        return model, tokenizer, device, None, None, None
    elif args.model.lower().startswith("roberta_"):
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name).cuda()
        device = "cuda"
        return model, tokenizer, device, None, None, None
    elif args.model.lower().startswith("distilbert_"):
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertModel.from_pretrained(model_name).cuda()
        device = "cuda"
        return model, tokenizer, device, None, None, None
    elif args.model.lower().startswith("xlnet_"):
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        model = XLNetModel.from_pretrained(model_name).cuda()
        device = "cuda"
        return model, tokenizer, device, None, None, None
    elif args.model.lower().startswith("xlm_"):
        tokenizer = XLMTokenizer.from_pretrained(model_name)
        model = XLMModel.from_pretrained(model_name).cuda()
        device = "cuda"
        return model, tokenizer, device, None, None, None

    if "phi3_small" in args.model.lower():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    if (
        "mistral" in args.model.lower()
        or "falcon" in args.model.lower()
        or "mpt" in args.model.lower()
        or "phi" in args.model.lower()
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
            token=args.access_token,
        )
    elif "pythia" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=args.revision,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=args.access_token,
        )
    else:
        revision = MODEL_DICT_LLMs[args.model].get("revision", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=args.access_token,
            revision=revision,
            trust_remote_code=True,
        )
    model.eval()

    if "mpt" in args.model.lower() or "pythia" in args.model.lower():
        if "seed" in model_name:
            model_name = model_name.rsplit("-", 1)[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=args.access_token)
    else:
        if "tinyllama" in args.model.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DICT_LLMs["tinyllama"]["model_id"], use_fast=False, token=args.access_token
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False, token=args.access_token, trust_remote_code=True
            )

    if "mpt_30b" in args.model.lower():
        device = model.hf_device_map["transformer.wte"]
    elif (
        "30b" in args.model.lower()
        or "65b" in args.model.lower()
        or "70b" in args.model.lower()
        or "40b" in args.model.lower()
    ):  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        # device = torch.device(str(model.hf_device_map["lm_head"]))
        device = torch.device("cuda:" + str(model.hf_device_map["lm_head"]))
    else:
        device = torch.device("cuda:" + str(model.hf_device_map[list(model.hf_device_map.keys())[-1]]))
        # device = torch.device("cuda")
    print(device)

    seq_len = 4096
    if (
        "lama" in args.model.lower()
        or "mistral" in args.model.lower()
        or "falcon3" in args.model.lower()
        or args.model.lower().startswith("qwen1.5_")
        or args.model.lower().startswith("qwen2_")
        or args.model.lower().startswith("qwen3_")
        or args.model.lower().startswith("qwen2.5_")
        or "vicuna" in args.model.lower()
    ):
        layers = model.model.layers
        hidden_size = model.config.hidden_size
        model.model.norm = torch.nn.Identity()
    elif "falcon" in args.model.lower():
        layers = model.transformer.h
        hidden_size = model.config.hidden_size
        model.transformer.ln_f = torch.nn.Identity()
    elif "mpt" in args.model.lower():
        layers = model.transformer.blocks
        hidden_size = model.config.d_model
        seq_len = 2048
        model.transformer.norm_f = torch.nn.Identity()
    elif "opt" in args.model.lower():
        layers = model.model.decoder.layers
        hidden_size = model.config.hidden_size
        seq_len = 2048
    elif "gpt2" in args.model.lower() or args.model.lower().startswith("qwen_"):
        layers = model.transformer.h
        hidden_size = model.transformer.embed_dim
        seq_len = 1024
        model.transformer.ln_f = torch.nn.Identity()
    elif "pythia" in args.model.lower():
        layers = model.gpt_neox.layers
        hidden_size = model.gpt_neox.config.hidden_size
        seq_len = 2048
        model.gpt_neox.final_layer_norm = torch.nn.Identity()
    elif "phi" in args.model.lower():
        layers = model.model.layers
        hidden_size = model.config.hidden_size
        if hasattr(model.model, "final_layernorm"):
            model.model.final_layernorm = torch.nn.Identity()
        else:
            model.model.norm = torch.nn.Identity()

    return model, tokenizer, device, layers, hidden_size, seq_len
