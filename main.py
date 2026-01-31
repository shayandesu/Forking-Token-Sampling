import os
import json
import math
import multiprocessing as mp
from utils import parse_args, get_dataset


def _worker(
    gpu_id: int | None,
    llm_kwargs: dict,
    df_shard,
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
):
    # Important: set device visibility BEFORE importing vLLM/torch.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPU visible → CPU

    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import VllmConfig
    from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate
    from vllm.v1.sample.logits_processor.builtin import process_dict_updates
    from vllm.sampling_params import RequestOutputKind

    class EntropyTempGate(LogitsProcessor):
        @classmethod
        def validate_params(cls, params: SamplingParams):
            extra = params.extra_args or {}
            enabled = extra.get("entropy_gate", False)
            if not isinstance(enabled, bool):
                raise ValueError("extra_args['entropy_gate'] must be bool")
            if enabled:
                h = extra.get("h_threshold", 0.672)
                t_ref = extra.get("t_ref", 1.0)  # temperature for entropy calc only
                t_high = extra.get("t_high", 1.5)
                t_low = extra.get("t_low", 0.7)
                for name, val in (("h_threshold", h), ("t_ref", t_ref), ("t_high", t_high), ("t_low", t_low)):
                    if not isinstance(val, (float, int)):
                        raise ValueError(f"{name} must be a number")
                if t_high <= 0 or t_low <= 0 or t_ref <= 0:
                    raise ValueError("t_ref, t_high, and t_low must be > 0")

        def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
            # req_info: request_id -> (h_threshold, t_ref, t_high, t_low)
            self.req_info: dict[int, tuple[float, float, float, float]] = {}

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            def extract(params: SamplingParams):
                self.validate_params(params)
                extra = params.extra_args or {}
                if not extra.get("entropy_gate", False):
                    return None
                return (
                    float(extra.get("h_threshold", 0.672)),
                    float(extra.get("t_ref", 1.0)),
                    float(extra.get("t_high", 1.5)),
                    float(extra.get("t_low", 0.7)),
                )

            process_dict_updates(
                self.req_info,
                batch_update,
                lambda params, _prompt, _output: extract(params),
            )

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            # logits: (num_requests, vocab_size)
            if not self.req_info:
                return logits

            rows = torch.tensor(list(self.req_info.keys()), device=logits.device, dtype=torch.long)
            h_thr = torch.tensor([self.req_info[i][0] for i in self.req_info.keys()],
                                 device=logits.device, dtype=torch.float32)
            t_ref = torch.tensor([self.req_info[i][1] for i in self.req_info.keys()],
                                device=logits.device, dtype=torch.float32)
            t_high = torch.tensor([self.req_info[i][2] for i in self.req_info.keys()],
                                  device=logits.device, dtype=torch.float32)
            t_low = torch.tensor([self.req_info[i][3] for i in self.req_info.keys()],
                                device=logits.device, dtype=torch.float32)

            sel = logits[rows].float()
            # Entropy at args.temperature (t_ref)
            logp = torch.log_softmax(sel / t_ref[:, None], dim=-1)
            p = torch.exp(logp)
            H = -(p * logp).sum(dim=-1)

            use_high = H > h_thr
            scale = torch.where(use_high, t_high, t_low)
            logits[rows] = (logits[rows].float() / scale[:, None]).to(logits.dtype)
            return logits
        
    llm_kwargs.update({
        "logits_processors": [EntropyTempGate],
    })

    llm = LLM(**llm_kwargs)

    # Process prompts (iterate over dataframe shard)
    for qid, row in df_shard.iterrows():
        prompt = prep(row["Problem"])
        all_outputs = []
        n_calls = math.ceil(samples / chunk_size)

        for c in range(n_calls):
            n = min(chunk_size, samples - c * chunk_size)
            if n <= 0:
                break

            sp = SamplingParams(
                n=n,
                max_tokens=max_tokens,
                temperature=1.0,  # gate sets effective temp via t_high/t_low; vLLM must not scale again
                top_p=top_p,
                seed=None if seed is None else (seed + c),
                output_kind=RequestOutputKind.FINAL_ONLY,
                extra_args={
                    "entropy_gate": True,
                    "h_threshold": h_threshold,
                    "t_ref": temperature,  # temperature for initial entropy calculation only
                    "t_high": t_high,
                    "t_low": t_low,
                },
            )

            res = llm.generate([prompt], [sp])[0]
            all_outputs.extend([o.text for o in res.outputs])

        out_queue.put({
            "id": qid,
            "prompt": prompt,
            "generations": all_outputs,
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
            }
        })

    out_queue.put(None)  # worker done sentinel


def main(args):
    # Resolve devices: CUDA_VISIBLE_DEVICES if set, else args.gpus; fallback to single CPU worker if none.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        # Use logical indices 0..n-1 for the n visible devices (workers will each see one).
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
        gpus = [None]  # single CPU worker
        
        
    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    # Load prompts
    df, prep = get_dataset(args.dataset_name)

    # Split df across workers (CPU: 1 worker, GPU: len(gpus) workers)
    n_workers = len(gpus)
    size = len(df)
    base, extra = divmod(size, n_workers)
    shards = []
    start = 0
    for i in range(n_workers):
        count = base + (1 if i < extra else 0)
        shards.append(df.iloc[start : start + count])
        start += count

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []

    for rank, gpu_id in enumerate(gpus):
        p = ctx.Process(
            target=_worker,
            args=(
                gpu_id,
                llm_kwargs,
                shards[rank],
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
            ),
        )
        p.start()
        procs.append(p)

    done = 0
    with open(args.out_path, "w", encoding="utf-8") as f_out:
        while done < len(procs):
            msg = q.get()
            if msg is None:
                done += 1
                continue
            f_out.write(json.dumps(msg, ensure_ascii=False) + "\n")
            f_out.flush()

    for p in procs:
        p.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)
