import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from serving_engine.request import RequestPhase

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


class ModelRunner:
    """Wraps the reused HF model/tokenizer (Decision 3: model internals are
    not the point, only the batching/memory orchestration around them is).
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        block_size: int,
        model_name: str = MODEL_NAME,
        device: str = "cuda",
        dtype=torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.block_size = block_size

        cfg = self.model.config
        self.kv_cache = torch.zeros(
            cfg.num_hidden_layers,
            2,  # K, V
            num_gpu_blocks,
            block_size,
            cfg.num_key_value_heads,
            cfg.head_dim,
            dtype=dtype,
            device=device,
        )

    def _flat_slot(self, block_table, logical_pos: int) -> int:
        """Physical flat position in self.kv_cache for a request's logical_pos-th token."""
        return block_table[logical_pos // self.block_size] * self.block_size + logical_pos % self.block_size

    def forward(self, scheduler_output) -> torch.Tensor:
        """Batched forward pass over a SchedulerOutput's requests."""
        # TODO: one combined call over all scheduled tokens (prefill and
        # decode mixed), attn_implementation="paged|eager" (reads/writes
        # self.kv_cache via a cache object's .update(), not the attention fn)
        # - read_index + attention_mask so each token only attends within
        #   its own request's valid range of the flattened batch
        # - reassemble logits by request_id, matching scheduled_requests'
        #   order (not by concatenation order)

        toks = []
        req_spans = []
        write_idxes = []
        for req in scheduler_output.scheduled_requests:
            start = len(toks)
            if req.phase == RequestPhase.NEEDS_PREFILL:
                toks += req.prompt_token_ids
                write_positions = range(req.prompt_len)
            else:  # NEEDS_DECODE
                toks.append(req.output_token_ids[-1])
                write_positions = [req.total_len - 1]

            for logical_pos in write_positions:
                write_idxes.append(self._flat_slot(req.block_table, logical_pos))
            req_spans.append((req, start, len(toks)))

        raise NotImplementedError
