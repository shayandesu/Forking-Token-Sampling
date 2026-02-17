import os, sys, argparse, json
from pathlib import Path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from post_processing.Gvendi.gradient_computer import GradientComputer
from post_processing.Gvendi.collect_gradients import get_model_and_tokenizer
from post_processing.Gvendi.gradient_vendi import GradientVendi


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str)
    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--proxy-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args(args=argv)
    return args


def get_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    ds_samples = {}
        
    for rec in records:
        qid = rec['id']
        generations = rec['generations']
        prompt = rec['prompt']
        if prompt.startswith('<|im_start|>'):
            prompt = prompt.split("<|im_start|>user")[-1].split("<|im_end|>")[0].strip()
        
        samples = []
        for gen in generations:
            if "</think>" in gen:
                gen = gen.split("</think>")[-1].strip()
            
            samples.append({
                "id": qid,
                "prompt": prompt,
                "completion": gen
            })
        
        ds_samples[qid] = samples
    
    return ds_samples
        

def main(args):
    out_path = args.out_path
    if not out_path:
        out_path = args.eval_path
    
    gen_path = Path(args.eval_path) / "generations.jsonl"
    
    ds = get_dataset(args.eval_path)
    
    out_path = Path(out_path)
        
    out_path = out_path / "g-vendi"    
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving to ", out_path)

    model, tokenizer = get_model_and_tokenizer(args.proxy_model)
    
    scores = {}
    for qid, samples in ds.items():
        out_file = out_path / f"{qid}"
        out_file.mkdir(parents=True, exist_ok=True)
        collector = GradientComputer(model_name=args.proxy_model, model=model, tokenizer=tokenizer)
        collector.compute_project_store_gradients(samples, out_file, 0)
        print(f"Processed Question {qid}. Computing G-Vendi.")
        sample_ids, sample_gradients = GradientVendi.load_all_gradients(out_file)
        g_vendi = GradientVendi.compute_gradient_vendi(sample_gradients)
        print(f"G-Vendi for {qid}: {g_vendi}")
        scores[qid] = g_vendi
    
    
    avg_score = sum([v for _, v in scores.items()]) / len(scores)
    
    scores['average'] = avg_score
    
    name = Path(args.eval_path).stem
    save_path = out_path / f"{name}_gvendi.jsonl"
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)