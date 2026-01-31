import os
import re
import json
import argparse

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
    """
    Extract the answer from model response, looking for content inside \\boxed{}
    """
    # Look for \boxed{answer} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, response)
    
    if matches:
        # Take the last \boxed{} content as the final answer
        # return [match.strip() for match in matches]
        return matches[-1].strip()
    
    # If no \boxed{} found, try to find the last number in the response
    # This is a fallback method
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        return numbers[-1]
    
    return None


def evaluate_correctness(predicted_answer, ground_truth):
    if predicted_answer is None:
        return False
    
    truth_clean = str(ground_truth).strip().lower()
    if isinstance(predicted_answer, list):
        pred_clean = [str(p_answer).strip().lower() for p_answer in predicted_answer]
        c_ness = [p == truth_clean for p in pred_clean]
        return any(c_ness)
    else:
        pred_clean = str(predicted_answer).strip().lower()
        
        if pred_clean == truth_clean:
            return True
    
    # Try to compare as numbers if possible
    # try:
    #     pred_num = float(pred_clean)
    #     truth_num = float(ground_truth)
    #     return abs(pred_num - truth_num) < 1e-10
    # except (ValueError, TypeError):
    #     pass
    
    return False



def evaluate_aime(df, output_json_path, k_values=(1, 4, 8, 16)):
    """
    Load main.py output JSONL, compute accuracy and pass@k, save results to a txt file
    next to output_json_path. df must have the same row order/index as the run (Answer column).
    """
    answer_col = "Answer" if "Answer" in df.columns else "answer"
    with open(output_json_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Per-problem: num_correct, num_samples
    results = []
    for rec in records:
        qid = rec["id"]
        generations = rec.get("generations", [])
        ground_truth = df.loc[qid, answer_col]
        num_correct = 0
        for text in generations:
            pred = extract_answer_from_response(text)
            if evaluate_correctness(pred, ground_truth):
                num_correct += 1
        results.append((num_correct, len(generations)))

    n_problems = len(results)
    if n_problems == 0:
        raise ValueError(f"No records in {output_json_path}")

    # Accuracy: fraction of problems with at least one correct generation
    accuracy = sum(1 for c, n in results if c > 0) / n_problems

    # pass@k (unbiased) averaged over problems
    pass_at_k = {}
    for k in k_values:
        pass_at_k[k] = sum(evaluate_pass_at_k(c, n, k) for c, n in results) / n_problems

    # Save next to input json file
    base, _ = os.path.splitext(output_json_path)
    out_txt_path = base + "_metrics.txt"
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation of {output_json_path}\n")
        f.write(f"# Problems: {n_problems}\n\n")
        f.write(f"accuracy: {accuracy:.4f}\n")
        for k in k_values:
            f.write(f"pass@{k}: {pass_at_k[k]:.4f}\n")

    return {"accuracy": accuracy, "pass_at_k": pass_at_k, "out_path": out_txt_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()
    
    if args.dataset_name not in ["aime2024", "aime2025"]:
        raise ValueError(f"Dataset {args.dataset_name} not supported for AIME evaluation")
    
    df, _ = get_dataset(args.dataset_name)
    results = evaluate_aime(df, args.out_path)
    print(results)
    

if __name__ == "__main__":
    main()