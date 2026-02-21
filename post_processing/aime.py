import os
import re
import json
import argparse
from pathlib import Path


try:
    from . import evaluate_pass_at_k
except ImportError:
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from post_processing import evaluate_pass_at_k

from utils import get_dataset


def extract_answer_from_response(response):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, response)
    if matches:
        return matches[-1].strip()
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        return numbers[-1]
    return None


def evaluate_correctness(predicted_answer, ground_truth):
    if predicted_answer is None:
        return False
    truth_clean = str(ground_truth).strip().lower()
    if isinstance(predicted_answer, list):
        pred_clean = [str(p).strip().lower() for p in predicted_answer]
        return any(p == truth_clean for p in pred_clean)
    else:
        return str(predicted_answer).strip().lower() == truth_clean


def evaluate_aime(df, eval_path, tokenizer=None, k_values=(1, 4, 8, 16, 100)):
    """
    Load main.py output JSONL, compute accuracy, pass@k, and shortest-correct-answer
    token counts. Saves results to eval_path/pass.jsonl.
    """
    answer_col = "Answer" if "Answer" in df.columns else "answer"
    if isinstance(eval_path, str):
        eval_path = Path(eval_path)

    gen_path = eval_path / "generations.jsonl"
    with open(gen_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    results = {}
    shortest_correct_token_counts = []  # one value per question (if any correct exists)

    for rec in records:
        qid = rec["id"]
        generations = rec.get("generations", [])
        ground_truth = df.loc[qid, answer_col]

        num_correct = 0
        shortest_correct_len = None  # token count of shortest correct answer

        for text in generations:
            pred = extract_answer_from_response(text)
            if evaluate_correctness(pred, ground_truth):
                num_correct += 1
                # Compute token length of this correct generation
                if tokenizer is not None:
                    tok_len = len(tokenizer.encode(text, add_special_tokens=False))
                else:
                    tok_len = len(text.split())  # fallback: whitespace tokens

                if shortest_correct_len is None or tok_len < shortest_correct_len:
                    shortest_correct_len = tok_len

        results[qid] = {
            "num_correct": num_correct,
            "num_samples": len(generations),
            "shortest_correct_tokens": shortest_correct_len,  # None if no correct answer
        }
        shortest_str = str(shortest_correct_len) if shortest_correct_len is not None else "N/A"
        print(f"{qid+1}: {num_correct}/{len(generations)}, shortest_correct_tokens={shortest_str}")

        if shortest_correct_len is not None:
            shortest_correct_token_counts.append(shortest_correct_len)

    n_problems = len(results)
    if n_problems == 0:
        raise ValueError(f"No records in {eval_path}")

    # Accuracy: fraction of problems with at least one correct generation
    accuracy = sum(1 for v in results.values() if v["num_correct"] > 0) / n_problems

    # pass@k (unbiased) averaged over problems
    pass_at_k = {}
    for k in k_values:
        pass_at_k[k] = sum(
            evaluate_pass_at_k(v["num_correct"], v["num_samples"], k)
            for v in results.values()
        ) / n_problems

    # Average shortest correct answer length
    avg_shortest_tokens = (
        sum(shortest_correct_token_counts) / len(shortest_correct_token_counts)
        if shortest_correct_token_counts else None
    )

    n_with_correct = len(shortest_correct_token_counts)
    print(f"\nProblems with at least one correct answer: {n_with_correct}/{n_problems}")
    if avg_shortest_tokens is not None:
        print(f"Avg shortest correct answer tokens: {avg_shortest_tokens:.1f}")

    # Build output dict
    output = {
        "accuracy": accuracy,
        "pass_at_k": {f"pass_at_{k}": v for k, v in pass_at_k.items()},
        "shortest_correct_tokens": {
            "per_question": {
                str(qid): v["shortest_correct_tokens"]
                for qid, v in results.items()
            },
            "average": avg_shortest_tokens,
            "n_questions_with_correct": n_with_correct,
        },
        "per_question": {
            str(qid): {
                "num_correct": v["num_correct"],
                "num_samples": v["num_samples"],
                "shortest_correct_tokens": v["shortest_correct_tokens"],
            }
            for qid, v in results.items()
        },
    }

    out_path = eval_path / "pass.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return {
        "accuracy": accuracy,
        "pass_at_k": pass_at_k,
        "avg_shortest_correct_tokens": avg_shortest_tokens,
        "out_path": out_path,
    }


def main(args):
    if args.dataset_name.lower() not in ["aime2024", "aime2025"]:
        raise ValueError(f"Dataset {args.dataset_name} not supported for AIME evaluation")

    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    df, _ = get_dataset(args.dataset_name)
    results = evaluate_aime(df, args.eval_path, tokenizer=tokenizer)
    print(results)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--eval-path", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name/path for tokenizer. "
                             "Falls back to whitespace splitting if not provided.")
    args = parser.parse_args(args=argv)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
