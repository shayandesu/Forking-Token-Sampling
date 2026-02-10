import torch
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm import SamplingParams

class EntropyTempGate(LogitsProcessor):
        @classmethod
        def validate_params(cls, params: SamplingParams):
            extra = params.extra_args or {}
            enabled = extra.get("entropy_gate", False)
            if not isinstance(enabled, bool):
                raise ValueError("extra_args['entropy_gate'] must be bool")
            if enabled:
                h = extra.get("h_threshold", 0.672)
                t_ref = extra.get("t_ref", 1.0)
                t_high = extra.get("t_high", 1.5)
                t_low = extra.get("t_low", 0.7)
                mode = extra.get("entropy_gate_mode", "temp")
                top_k = extra.get("entropy_gate_top_k", 5)

                for name, val in (("h_threshold", h), ("t_ref", t_ref), ("t_high", t_high), ("t_low", t_low)):
                    if not isinstance(val, (float, int)):
                        raise ValueError(f"{name} must be a number")
                if t_high <= 0 or t_low <= 0 or t_ref <= 0:
                    raise ValueError("t_ref, t_high, and t_low must be > 0")
                if mode not in ("temp", "topk"):
                    raise ValueError("entropy_gate_mode must be 'temp' or 'topk'")
                if not isinstance(top_k, int) or top_k < 1:
                    raise ValueError("entropy_gate_top_k must be a positive integer")

        def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
            self.req_info: dict[int, tuple[float, float, float, float, str, int]] = {}

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
                    str(extra.get("entropy_gate_mode", "temp")),
                    int(extra.get("entropy_gate_top_k", 5)),
                )

            process_dict_updates(
                self.req_info,
                batch_update,
                lambda params, _prompt, _output: extract(params),
            )

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            rows = torch.tensor(list(self.req_info.keys()), device=logits.device, dtype=torch.long)
            infos = [self.req_info[i] for i in self.req_info.keys()]

            h_thr = torch.tensor([x[0] for x in infos], device=logits.device, dtype=torch.float32)
            t_ref = torch.tensor([x[1] for x in infos], device=logits.device, dtype=torch.float32)
            t_high = torch.tensor([x[2] for x in infos], device=logits.device, dtype=torch.float32)
            t_low = torch.tensor([x[3] for x in infos], device=logits.device, dtype=torch.float32)
            modes = [x[4] for x in infos]
            top_ks = [x[5] for x in infos]

            sel = logits[rows].float()

            logp = torch.log_softmax(sel / t_ref[:, None], dim=-1)
            p = torch.exp(logp)
            H = -(p * logp).sum(dim=-1)

            # Collect entropy values (detach + CPU immediately to avoid GPU memory issues)
            collector = getattr(EntropyTempGate, "_entropy_collector", None)
            if collector is not None:
                for val in H.detach().cpu().tolist():
                    collector.append(val)

            # Process each row
            out = logits[rows].float().clone()
            for idx, (use_high, mode, k) in enumerate(zip(H > h_thr, modes, top_ks)):
                if mode == "temp":
                    scale = t_high[idx] if use_high else t_low[idx]
                    out[idx] = out[idx] / scale
                else:  # mode == "topk"
                    if use_high:
                        _, top_indices = torch.topk(sel[idx], min(k, sel.shape[1]))
                        mask = torch.full_like(out[idx], float("-inf"))
                        mask[top_indices] = 0.0
                        out[idx] = mask
                    else:
                        out[idx] = out[idx] / t_low[idx]

            logits[rows] = out.to(logits.dtype)

            # Record forking tokens
            high = (H > h_thr).sum().item()
            stats = getattr(EntropyTempGate, "_forking_stats", None)
            if stats is not None:
                stats[0] = stats[0] + high
                stats[1] = stats[1] + H.numel()

            return logits