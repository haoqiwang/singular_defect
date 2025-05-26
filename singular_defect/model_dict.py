CACHE_DIR_BASE = "./model_weights"

MODEL_DICT_LLMs = {
    ### bert model
    "bert_base_uncased": {"model_id": "google-bert/bert-base-uncased", "cache_dir": CACHE_DIR_BASE},
    "bert_base_cased": {"model_id": "google-bert/bert-base-cased", "cache_dir": CACHE_DIR_BASE},
    "bert_large_uncased": {"model_id": "google-bert/bert-large-uncased", "cache_dir": CACHE_DIR_BASE},
    "bert_large_cased": {"model_id": "google-bert/bert-large-cased", "cache_dir": CACHE_DIR_BASE},
    "bert_base_chinese": {"model_id": "google-bert/bert-base-chinese", "cache_dir": CACHE_DIR_BASE},
    ### roberta model
    "roberta_base": {"model_id": "FacebookAI/roberta-base", "cache_dir": CACHE_DIR_BASE},
    "roberta_large": {"model_id": "FacebookAI/roberta-large", "cache_dir": CACHE_DIR_BASE},
    ### distillbert model
    "distilbert_base_uncased": {"model_id": "distilbert/distilbert-base-uncased", "cache_dir": CACHE_DIR_BASE},
    "distilbert_base_cased": {"model_id": "distilbert/distilbert-base-cased", "cache_dir": CACHE_DIR_BASE},
    ### xlnet model
    "xlnet_base_cased": {"model_id": "xlnet/xlnet-base-cased", "cache_dir": CACHE_DIR_BASE},
    "xlnet_large_cased": {"model_id": "xlnet/xlnet-large-cased", "cache_dir": CACHE_DIR_BASE},
    ### xlm model
    "xlm_mlm_ende": {"model_id": "FacebookAI/xlm-mlm-ende-1024", "cache_dir": CACHE_DIR_BASE},
    "xlm_clm_ende": {"model_id": "FacebookAI/xlm-clm-ende-1024", "cache_dir": CACHE_DIR_BASE},
    ### qwen3 model
    "qwen3_0.6b": {"model_id": "Qwen/Qwen3-0.6B", "cache_dir": CACHE_DIR_BASE},
    ### llama2 model
    "llama2_7b": {"model_id": "meta-llama/Llama-2-7b-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_13b": {"model_id": "meta-llama/Llama-2-13b-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_70b": {"model_id": "meta-llama/Llama-2-70b-hf", "cache_dir": CACHE_DIR_BASE},
    ### llama2 chat model
    "llama2_7b_chat": {"model_id": "meta-llama/Llama-2-7b-chat-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_13b_chat": {"model_id": "meta-llama/Llama-2-13b-chat-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_70b_chat": {"model_id": "meta-llama/Llama-2-70b-chat-hf", "cache_dir": CACHE_DIR_BASE},
    ### llama2 code model
    "llama2_7b_code": {"model_id": "meta-llama/CodeLlama-7b-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_7b_code_python": {"model_id": "meta-llama/CodeLlama-7b-Python-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_7b_code_instruct": {"model_id": "meta-llama/CodeLlama-7b-Instruct-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_13b_code": {"model_id": "meta-llama/CodeLlama-13b-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_13b_code_python": {"model_id": "meta-llama/CodeLlama-13b-Python-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_13b_code_instruct": {"model_id": "meta-llama/CodeLlama-13b-Instruct-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_34b_code": {"model_id": "meta-llama/CodeLlama-34b-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_34b_code_python": {"model_id": "meta-llama/CodeLlama-34b-Python-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_34b_code_instruct": {"model_id": "meta-llama/CodeLlama-34b-Instruct-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_70b_code": {"model_id": "meta-llama/CodeLlama-70b-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_70b_code_python": {"model_id": "meta-llama/CodeLlama-70b-Python-hf", "cache_dir": CACHE_DIR_BASE},
    "llama2_70b_code_instruct": {"model_id": "meta-llama/CodeLlama-70b-Instruct-hf", "cache_dir": CACHE_DIR_BASE},
    ### llama3 model
    "llama3_8b": {"model_id": "meta-llama/Meta-Llama-3-8B", "cache_dir": CACHE_DIR_BASE},
    "llama3_8b_instruct": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "cache_dir": CACHE_DIR_BASE},
    "llama3_8b_guard": {"model_id": "meta-llama/Meta-Llama-Guard-2-8B", "cache_dir": CACHE_DIR_BASE},
    "llama3_70b": {"model_id": "meta-llama/Meta-Llama-3-70B", "cache_dir": CACHE_DIR_BASE},
    "llama3_70b_instruct": {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct", "cache_dir": CACHE_DIR_BASE},
    ### llama3.1 model
    "llama3.1_8b": {"model_id": "meta-llama/Llama-3.1-8B", "cache_dir": CACHE_DIR_BASE},
    "llama3.1_8b_instruct": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "cache_dir": CACHE_DIR_BASE},
    "llama3.1_8b_guard": {"model_id": "meta-llama/Llama-Guard-3-8B", "cache_dir": CACHE_DIR_BASE},
    "llama3.1_70b": {"model_id": "meta-llama/Llama-3.1-70B", "cache_dir": CACHE_DIR_BASE},
    "llama3.1_70b_instruct": {"model_id": "meta-llama/Llama-3.1-70B-Instruct", "cache_dir": CACHE_DIR_BASE},
    ### llama3.2 model
    "llama3.2_1b": {"model_id": "meta-llama/Llama-3.2-1B", "cache_dir": CACHE_DIR_BASE},
    "llama3.2_1b_instruct": {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "cache_dir": CACHE_DIR_BASE},
    "llama3.2_1b_guard": {"model_id": "meta-llama/Llama-Guard-3-1B", "cache_dir": CACHE_DIR_BASE},
    "llama3.2_3b": {"model_id": "meta-llama/Llama-3.2-3B", "cache_dir": CACHE_DIR_BASE},
    "llama3.2_3b_instruct": {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "cache_dir": CACHE_DIR_BASE},
    ### llama3.3 model
    "llama3.3_70b_instruct": {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "cache_dir": CACHE_DIR_BASE},
    ### deepseek model
    "deepseek_r1_llama_8b": {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "cache_dir": CACHE_DIR_BASE},
    ### bert model
    "bert_large": {"model_id": "google-bert/bert-large-cased-whole-word-masking", "cache_dir": CACHE_DIR_BASE},
    ### mistral model
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_moe": {
        "model_id": "mistralai/Mixtral-8x7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_7b_instruct": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral2_7b_instruct": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral3_7b_instruct": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_moe_instruct": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral3_7b": {
        "model_id": "mistralai/Mistral-7B-v0.3",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### phi-1
    "phi1": {
        "model_id": "microsoft/phi-1",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### phi-1.5
    "phi15": {
        "model_id": "microsoft/phi-1_5",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### phi-2
    "phi2": {
        "model_id": "microsoft/phi-2",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### phi-3
    "phi3_mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "phi3_mini_128k": {
        "model_id": "microsoft/Phi-3-mini-128k-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "phi3_small": {
        "model_id": "microsoft/Phi-3-small-8k-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "phi3_small_128k": {
        "model_id": "microsoft/Phi-3-small-128k-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "phi3_medium": {
        "model_id": "microsoft/Phi-3-medium-4k-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "phi3_medium_128k": {
        "model_id": "microsoft/Phi-3-medium-128k-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### phi3.5
    "phi3.5_mini": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "phi3.5_moe": {
        "model_id": "microsoft/Phi-3.5-MoE-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### falcon model
    "falcon_7b": {
        "model_id": "tiiuae/falcon-7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon_7b_instruct": {
        "model_id": "tiiuae/falcon-7b-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon_40b": {
        "model_id": "tiiuae/falcon-40b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon2_11b": {
        "model_id": "tiiuae/falcon-11B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon3_1b": {
        "model_id": "tiiuae/Falcon3-1B-Base",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon3_1b_instruct": {
        "model_id": "tiiuae/Falcon3-1B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon3_3b": {
        "model_id": "tiiuae/Falcon3-3B-Base",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon3_3b_instruct": {
        "model_id": "tiiuae/Falcon3-3B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon3_7b": {
        "model_id": "tiiuae/Falcon3-7B-Base",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon3_7b_instruct": {
        "model_id": "tiiuae/Falcon3-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon3_10b": {
        "model_id": "tiiuae/Falcon3-10B-Base",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon3_10b_instruct": {
        "model_id": "tiiuae/Falcon3-10B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### qwen model
    "qwen_1.8b": {
        "model_id": "Qwen/Qwen-1_8B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen_1.8b_chat": {
        "model_id": "Qwen/Qwen-1_8B-Chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen_7b": {
        "model_id": "Qwen/Qwen-7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen_7b_chat": {
        "model_id": "Qwen/Qwen-7B-Chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_0.5b": {
        "model_id": "Qwen/Qwen1.5-0.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_0.5b_chat": {
        "model_id": "Qwen/Qwen1.5-0.5B-Chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_1.8b": {
        "model_id": "Qwen/Qwen1.5-1.8B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_1.8b_chat": {
        "model_id": "Qwen/Qwen1.5-1.8B-Chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_4b": {
        "model_id": "Qwen/Qwen1.5-4B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_4b_chat": {
        "model_id": "Qwen/Qwen1.5-4B-Chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_7b": {
        "model_id": "Qwen/Qwen1.5-7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_7b_chat": {
        "model_id": "Qwen/Qwen1.5-7B-Chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_0.5b": {
        "model_id": "Qwen/Qwen2-0.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_0.5b_instruct": {
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_1.5b": {
        "model_id": "Qwen/Qwen2-1.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_1.5b_instruct": {
        "model_id": "Qwen/Qwen2-1.5B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_1.5b_math": {
        "model_id": "Qwen/Qwen2-Math-1.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_1.5b_math_instruct": {
        "model_id": "Qwen/Qwen2-Math-1.5B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_7b": {
        "model_id": "Qwen/Qwen2-7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_7b_instruct": {
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_7b_math": {
        "model_id": "Qwen/Qwen2-Math-7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_7b_math_instruct": {
        "model_id": "Qwen/Qwen2-Math-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_0.5b_instruct": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_0.5b_code": {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_0.5b_code_instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_1.5b": {
        "model_id": "Qwen/Qwen2.5-1.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_1.5b_instruct": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_1.5b_math": {
        "model_id": "Qwen/Qwen2.5-Math-1.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_1.5b_math_instruct": {
        "model_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_1.5b_code": {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_1.5b_code_instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_3b": {
        "model_id": "Qwen/Qwen2.5-3B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_3b_instruct": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_3b_code": {
        "model_id": "Qwen/Qwen2.5-Coder-3B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_3b_code_instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_7b": {
        "model_id": "Qwen/Qwen2.5-7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_7b_instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_7b_math": {
        "model_id": "Qwen/Qwen2.5-Math-7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_7b_math_instruct": {
        "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_7b_code": {
        "model_id": "Qwen/Qwen2.5-Coder-7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2.5_7b_code_instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### vicuna model
    "vicuna1.1_7b": {
        "model_id": "lmsys/vicuna-7b-v1.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "vicuna1.3_7b": {
        "model_id": "lmsys/vicuna-7b-v1.3",
        "cache_dir": CACHE_DIR_BASE,
    },
    "vicuna1.5_7b": {
        "model_id": "lmsys/vicuna-7b-v1.5",
        "cache_dir": CACHE_DIR_BASE,
    },
    "vicuna1.5_7b_16k": {
        "model_id": "lmsys/vicuna-7b-v1.5-16k",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### mpt model
    "mpt_7b": {
        "model_id": "mosaicml/mpt-7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mpt_7b_instruct": {
        "model_id": "mosaicml/mpt-7b-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mpt_7b_chat": {
        "model_id": "mosaicml/mpt-7b-chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mpt_7b_storywriter": {
        "model_id": "mosaicml/mpt-7b-storywriter",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mpt_30b": {
        "model_id": "mosaicml/mpt-30b",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### opt model
    "opt_125m": {
        "model_id": "facebook/opt-125m",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_350m": {
        "model_id": "facebook/opt-350m",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_1.3b": {
        "model_id": "facebook/opt-1.3b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_2.7b": {
        "model_id": "facebook/opt-2.7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_7b": {
        "model_id": "facebook/opt-6.7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_13b": {
        "model_id": "facebook/opt-13b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_30b": {
        "model_id": "facebook/opt-30b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_66b": {
        "model_id": "facebook/opt-66b",
        "cache_dir": CACHE_DIR_BASE,
    },
    ### gpt2 model
    "gpt2": {"model_id": "gpt2", "cache_dir": CACHE_DIR_BASE},
    "gpt2_medium": {"model_id": "gpt2-medium", "cache_dir": CACHE_DIR_BASE},
    "gpt2_large": {"model_id": "gpt2-large", "cache_dir": CACHE_DIR_BASE},
    "gpt2_xl": {"model_id": "gpt2-xl", "cache_dir": CACHE_DIR_BASE},
}


for scale in ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]:
    MODEL_DICT_LLMs[f"pythia_{scale}"] = {
        "model_id": f"EleutherAI/pythia-{scale}",
        "cache_dir": CACHE_DIR_BASE,
    }

for scale in ["14m", "31m", "70m", "160m", "410m"]:
    for seed in range(1, 10):
        MODEL_DICT_LLMs[f"pythia_{scale}_{seed}"] = {
            "model_id": f"EleutherAI/pythia-{scale}-seed{seed}",
            "cache_dir": CACHE_DIR_BASE,
        }
