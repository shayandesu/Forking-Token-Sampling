import pandas as pd
import argparse
from datasets import load_dataset


def build_prompt(question):
    return (
        "Please reason step by step, and put your final answer within \\boxed{}\n"
        f"{question}"
    )

def get_dataset(name):
    ds_name = name.lower().strip()
    
    if ds_name == "aime2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024")
        df = ds["train"].to_pandas()
        prep = build_prompt
    elif ds_name =="aime2025":
        df1 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test'].to_pandas()
        df2 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test'].to_pandas()
        df = pd.concat([df1, df2])
        df.rename(columns={"question": "Problem", "answer": "Answer"}, inplace=True)
        prep = build_prompt
    elif ds_name == "math500":
        df = load_dataset("HuggingFaceH4/MATH-500")['test'].to_pandas()
        df.rename(columns={"problem": "Problem", "answer": "Answer"}, inplace=True)
        prep = build_prompt
    else:
        try:
            df = load_dataset(name).to_pandas()
        except:
            raise NameError(f"Dataset {name} not available.")
    
    return df, prep

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B-AWQ")
    ap.add_argument('-d', "--dataset-name", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    ap.add_argument("-q", "--quantization", type=str, default=None, choices=[None, "bitsandbytes", "fp8"])
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--no_tqdm", action="store_true", help="Disable vLLM tqdm in generate()")
    

    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--h-threshold", type=float, default=0.672)
    ap.add_argument("--t-high", type=float, default=2.0)
    ap.add_argument("--t-low", type=float, default=0.7)
    ap.add_argument(
        "--entropy-gate-mode",
        type=str,
        default="temp",
        choices=["temp", "topk"],
        help="When entropy > threshold: 'temp' = sample with higher temperature; 'topk' = choose uniformly from top-k tokens",
    )
    ap.add_argument(
        "--entropy-gate-top-k",
        type=int,
        default=5,
        help="For entropy-gate-mode=topk: number of top candidate tokens to sample from uniformly",
    )

    ap.add_argument("--gpus", type=str, default="all")
    args = ap.parse_args()
    
    return args