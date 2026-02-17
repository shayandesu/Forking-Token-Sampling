import os
import json
import math
import multiprocessing as mp
import tempfile
from utils import parse_args, get_dataset
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from transformers import AutoTokenizer

def _write_results(results: dict, df, out_path: Path) -> None:
    """Atomically write merged results to out_path in df.index order."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False, dir=out_path.parent
    ) as f_out:
        tmp_path = Path(f_out.name)
        for qid in df.index:
            if qid in results:
                f_out.write(json.dumps(results[qid], ensure_ascii=False) + "\n")
        tmp_path.replace(out_path)

def _load_existing(out_path: Path, samples: int) -> dict:
    """Load existing output file. Returns {qid: record} for completed and incomplete questions."""
    result = {}
    if not out_path.exists():
        return result
    
    print("File Already Exists. Checking the file...")

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            print(line[:100])
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("id")
            if qid is None:
                continue
            result[qid] = rec

    return result

def _worker(
    gpu_id: int | None,
    llm_kwargs: dict,
    todo_shard: list,
    out_queue: mp.Queue,
    prep: callable,
    samples: int,
    chunk_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    h_threshold: float,
    t_high: float,
    t_low: float,
    entropy_gate_mode: str,
    entropy_gate_top_k: int,
    no_tqdm: bool,
):
    # Important: set device visibility BEFORE importing vLLM/torch.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPU visible → CPU

    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from logit_processor import EntropyTempGate

    # Shared state for stats
    _mgr = mp.Manager()
    forking_stats = _mgr.list([0, 0])
    entropy_collector = _mgr.list()  # collect H values (CPU floats) during generation

    EntropyTempGate._forking_stats = forking_stats
    EntropyTempGate._entropy_collector = entropy_collector

    llm_kwargs.update({"logits_processors": [EntropyTempGate]})
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Process prompts
    for qid, problem_text, existing_gens in todo_shard:
        prompt = prep(problem_text)
        prompt = tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True
        )
        all_outputs = list(existing_gens) if existing_gens else []
        curr_h_threshold = float(h_threshold)

        # Clear entropy collector for this qid
        entropy_collector[:] = []

        samples_needed = samples - len(all_outputs)
        n_calls = math.ceil(samples_needed / chunk_size) if samples_needed > 0 else 0
        chunk_offset = len(all_outputs) // chunk_size
        # perplexities = []
        
        ent_vals_all = []

        for c in range(n_calls):
            n = min(chunk_size, samples_needed - c * chunk_size)
            if n <= 0:
                break

            forking_stats[0] = forking_stats[1] = 0
            
            entropy_collector[:] = []

            sp = SamplingParams(
                n=n,
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=1.0 if entropy_gate_mode == "topk" else top_p,
                seed=None if seed is None else (seed + chunk_offset + c),
                output_kind=RequestOutputKind.FINAL_ONLY,
                # logprobs=1,
                extra_args={
                    "entropy_gate": True,
                    "h_threshold": curr_h_threshold,
                    "t_ref": temperature,
                    "t_high": t_high,
                    "t_low": t_low,
                    "entropy_gate_mode": entropy_gate_mode,
                    "entropy_gate_top_k": entropy_gate_top_k,
                },
            )

            res = llm.generate([prompt], [sp], use_tqdm=not no_tqdm)[0]
            all_outputs.extend([o.text for o in res.outputs])
            
            ent_vals_chunk = list(entropy_collector)
            if not ent_vals_chunk:
                raise ValueError("No entropy values collected for this chunk")
            
            import torch as torch_local
            ent_t = torch_local.tensor(ent_vals_chunk, dtype=torch_local.float32)
            chunk_p80 = float(torch_local.quantile(ent_t, 0.8).item())

            curr_h_threshold = 0.9 * curr_h_threshold + 0.1 * chunk_p80
            ent_vals_all.extend(ent_vals_chunk)


            n_high, n_total = forking_stats[0], forking_stats[1]
            pct = 100.0 * n_high / n_total if n_total > 0 else 0.0

            out_queue.put({
                "type": "progress",
                "gpu": gpu_id,
                "qid": int(qid),
                "call": c + 1,
                "total": n_calls,
                "forking_count": n_high,
                "forking_pct": pct,
                "token_count": n_total,
                "curr_thresh": curr_h_threshold
            })

        # Compute entropy stats for this qid
        ent_vals = list(entropy_collector)
        if ent_vals:
            import torch as torch_local
            ent_t = torch_local.tensor(ent_vals, dtype=torch_local.float32)
            ent_mean = float(ent_t.mean().item())
            ent_p80 = float(torch_local.quantile(ent_t, 0.8).item())
            ent_n = len(ent_vals)
        else:
            raise ValueError("Problem fetching entropy values")
        
        # avg_ppl = sum(perplexities) / len(perplexities)

        # Send entropy stats
        out_queue.put({
            "type": "entropy_stats",
            "gpu": gpu_id,
            "qid": int(qid),
            "entropy_avg": ent_mean,
            "entropy_p80": ent_p80,
            "entropy_n": ent_n,
            "entropy_values": ent_vals,  # send raw values for global aggregation
            # "perplexity": avg_ppl,
        })

        # Send final result for this qid
        out_queue.put({
            "id": qid,
            "prompt": prompt,
            "generations": all_outputs,
            # "perplexities": perplexities,
            "meta": {
                "gpu": gpu_id,
                "samples": samples,
                "chunk_size": chunk_size,
                "temperature": temperature,
                "top_p": top_p,
                "h_threshold": h_threshold,
                "t_ref": temperature,
                "t_high": t_high,
                "t_low": t_low,
                "entropy_avg": ent_mean,
                "entropy_p80": ent_p80,
                # "perplexity_avg": avg_ppl,
            }
        })

    out_queue.put(None)  # worker done sentinel

def main(args):
    pprint(vars(args), sort_dicts=False)

    # Resolve devices
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        gpus = list(range(len([x for x in cvd.split(",") if x.strip()])))
    else:
        gpu_arg = (args.gpus or "").strip().lower()
        if gpu_arg == "all":
            import torch
            n = torch.cuda.device_count()
            gpus = list(range(n)) if n > 0 else []
        else:
            gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    if not gpus:
        gpus = [None]

    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    # Load prompts
    df, prep = get_dataset(args.dataset_name)
    mode = (f"temp_{args.temperature:.1f}_{args.t_high:.1f}_{args.t_low:.1f}"
            if args.entropy_gate_mode == "temp"
            else f"topk_{args.entropy_gate_top_k}")
    
    file_name = f"{args.dataset_name.upper()}/{args.model.split("/")[-1]}_{mode}_{args.max_tokens}_{args.seed}.jsonl"
    full_path = os.path.join(args.out_dir, file_name)
    # assert not os.path.exists(full_path)
    print(f"Saving results to {full_path}")
    out_path = Path(full_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output
    existing = _load_existing(out_path, args.samples)
    results: dict = {qid: existing[qid] for qid in existing if len(existing[qid].get("generations", [])) >= args.samples}

    # Build todo shards
    n_workers = len(gpus)
    size = len(df)
    base, extra = divmod(size, n_workers)
    todo_shards = [[] for _ in range(n_workers)]
    start = 0

    for i in range(n_workers):
        count = base + (1 if i < extra else 0)
        for j in range(count):
            qid = df.index[start + j]
            if qid in results:
                continue
            rec = existing.get(qid, {})
            existing_gens = rec.get("generations", [])
            problem_text = df.loc[qid, "Problem"]
            todo_shards[i].append((qid, problem_text, existing_gens))
        start += count

    n_todo = sum(len(s) for s in todo_shards)

    if n_todo == 0:
        print(f"All {size} questions already completed. Nothing to do.", flush=True)
        return

    if results:
        print(f"Resuming: {len(results)} completed, {n_todo} to process.", flush=True)
    else:
        print(f"Starting: {n_todo} questions to process.", flush=True)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []

    for rank, gpu_id in enumerate(gpus):
        p = ctx.Process(
            target=_worker,
            args=(
                gpu_id,
                llm_kwargs,
                todo_shards[rank],
                q,
                prep,
                args.samples,
                args.chunk_size,
                args.max_tokens,
                args.temperature,
                args.top_p,
                None if args.seed is None else (args.seed + 100000 * rank),
                args.h_threshold,
                args.t_high,
                args.t_low,
                args.entropy_gate_mode,
                args.entropy_gate_top_k,
                args.no_tqdm,
            ),
        )
        p.start()
        procs.append(p)

    done = 0
    global_entropy_values = []

    while done < len(procs):
        msg = q.get()

        if msg is None:
            done += 1
            continue

        if msg.get("type") == "progress":
            fc = msg.get("forking_count", 0)
            fp = msg.get("forking_pct", 0.0)
            tc = msg.get("token_count", 0)
            cth = msg.get("curr_thresh", 0)
            print(
                f"[gpu {msg['gpu']}] qid={msg['qid']} {msg['call']}/{msg['total']} "
                f"— forking tokens: {fc}/{tc} ({fp:.1f}%), current threshold: {cth:.3f}",
                flush=True,
            )
            continue

        if msg.get("type") == "entropy_stats":
            avg = msg.get("entropy_avg")
            p80 = msg.get("entropy_p80")
            n = msg.get("entropy_n", 0)
            # avg_ppl = msg.get("preplexity", 0)

            avg_s = f"{avg:.4f}" if avg is not None else "N/A"
            p80_s = f"{p80:.4f}" if p80 is not None else "N/A"

            print(
                f"[gpu {msg['gpu']}] qid={msg['qid']} "
                f"entropy(avg={avg_s}, p80={p80_s}, n={n})",
                flush=True,
            )

            # Accumulate entropy values for global stats
            ent_vals = msg.get("entropy_values", [])
            if ent_vals:
                global_entropy_values.extend(ent_vals)

            continue

        results[msg["id"]] = msg
        _write_results(results, df, out_path)

    for p in procs:
        p.join()

    # Print global entropy stats
    if global_entropy_values:
        import torch
        global_ent = torch.tensor(global_entropy_values, dtype=torch.float32)
        global_avg = float(global_ent.mean().item())
        global_p80 = float(torch.quantile(global_ent, 0.8).item())
        global_n = len(global_entropy_values)

        print("\n" + "="*60, flush=True)
        print(
            f"GLOBAL ENTROPY STATS: avg={global_avg:.4f}, p80={global_p80:.4f}, n={global_n}",
            flush=True,
        )
        print("="*60 + "\n", flush=True)
    else:
        print("\nNo entropy values collected.\n", flush=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
