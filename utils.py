import pandas as pd
import argparse
from datasets import load_dataset


def build_prompt_aime(question):
    return [
        {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{question}"}
    ]
    
def build_prompt_lcb(question):
    return (
        "You are an expert Python programmer.\n"
        "You will be given a programming problem and must generate a correct "
        "Python solution that matches the specification and passes all "
        "tests.\n\n"
        f"{question}\n\n"
        "Format:\n"
        "You will use the following starter code to write the solution "
        "and enclose your code within backticks."
        "```python\n"
        "class Solution:\n"
        "    def solve(self, ...):\n"
        "        pass\n\n"
        "Answer:"
    )
    

def get_dataset(name):
    ds_name = name.lower().strip()
    
    if ds_name == "aime2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024")
        df = ds["train"].to_pandas()
        prep = build_prompt_aime
    elif ds_name =="aime2025":
        df1 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test'].to_pandas()
        df2 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test'].to_pandas()
        df = pd.concat([df1, df2])
        df.rename(columns={"question": "Problem", "answer": "Answer"}, inplace=True)
        prep = build_prompt_aime
    elif ds_name == "math500":
        df = load_dataset("HuggingFaceH4/MATH-500")['test'].to_pandas()
        df.rename(columns={"problem": "Problem", "answer": "Answer"}, inplace=True)
        prep = build_prompt_aime
    elif ds_name == "lcb":
        ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")
        df = ds["test"].to_pandas()
        df = df[df["difficulty"] == "hard"]
        prep = build_prompt_lcb
    else:
        try:
            df = load_dataset(name).to_pandas()
        except:
            raise NameError(f"Dataset {name} not available.")
    
    return df, prep

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument('-d', "--dataset-name", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("-q", "--quantization", type=str, default=None, choices=[None, "bitsandbytes", "fp8"])
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--no-tqdm", action="store_true", help="Disable vLLM tqdm in generate()")
    

    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--h-threshold", type=float, default=0.672)
    ap.add_argument("--t-high", type=float, default=1.4)
    ap.add_argument("--t-low", type=float, default=0.6)
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
    args = ap.parse_args(args=argv)
    
    return args