import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from serving_engine.request import RequestPhase

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


class _PagedKVCache:
    """Duck-typed cache object: LlamaAttention.forward() calls .update() once
    per layer, per forward pass. Wraps ModelRunner's physical self.kv_cache."""

    def __init__(self, kv_cache: torch.Tensor):
        self.kv_cache = kv_cache  # [num_layers, 2, num_gpu_blocks, block_size, num_kv_heads, head_dim]
        self.num_kv_heads = kv_cache.shape[-2]
        self.head_dim = kv_cache.shape[-1]

    def update(self, key_states, value_states, layer_idx, read_index, write_index):
        key_states = key_states.transpose(1, 2).squeeze(0)  # [seqlen, num_kv_heads, head_dim]
        value_states = value_states.transpose(1, 2).squeeze(0)

        k_cache = self.kv_cache[layer_idx, 0].view(-1, self.num_kv_heads, self.head_dim)
        v_cache = self.kv_cache[layer_idx, 1].view(-1, self.num_kv_heads, self.head_dim)

        k_cache.index_copy_(0, write_index, key_states)
        v_cache.index_copy_(0, write_index, value_states)

        return torch.index_select(k_cache, 0, read_index), torch.index_select(v_cache, 0, read_index)


class ModelRunner:
    """Wraps the reused HF model/tokenizer (Decision 3: model internals are
    not the point, only the batching/memory orchestration around them is).
    """

    def __init__(
        self,
        num_gpu_blocks: int | None,
        block_size: int,
        model_name: str = MODEL_NAME,
        device: str = "cuda",
        dtype=torch.bfloat16,
        gpu_memory_utilization: float = 0.9,
    ):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(device)
        self.model.eval()
        self.model.set_attn_implementation("paged|eager")
        self.block_size = block_size

        cfg = self.model.config
        # Free memory must be read here: after weights are loaded (a fixed
        # baseline that has to be subtracted first) but before the cache
        # tensor below claims any of it.
        if num_gpu_blocks is None:
            num_gpu_blocks = self._infer_num_gpu_blocks(cfg, dtype, gpu_memory_utilization)
        self.num_gpu_blocks = num_gpu_blocks

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

    def _infer_num_gpu_blocks(self, cfg, dtype, gpu_memory_utilization: float) -> int:
        """Size the KV cache pool from actual free GPU memory, holding back
        (1 - gpu_memory_utilization) as headroom for activation memory."""
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        usable_bytes = int(free_bytes * gpu_memory_utilization)
        bytes_per_block = (
            2  # K, V
            * cfg.num_hidden_layers
            * self.block_size
            * cfg.num_key_value_heads
            * cfg.head_dim
            * torch.empty((), dtype=dtype).element_size()
        )
        return usable_bytes // bytes_per_block

    def _flat_slot(self, block_table, logical_pos: int) -> int:
        """Physical flat position in self.kv_cache for a request's logical_pos-th token."""
        return block_table[logical_pos // self.block_size] * self.block_size + logical_pos % self.block_size

    @torch.inference_mode()
    def forward(self, scheduler_output) -> torch.Tensor:
        """Batched forward pass over a SchedulerOutput's requests: one
        combined call mixing prefill and decode tokens, gathering/scattering
        self.kv_cache via _PagedKVCache.update() inside each attention layer."""

        toks = []
        req_spans = []
        write_idxes = []
        read_idxes = []
        position_ids = []  # per query token, logical position within its own request
        query_group = []   # per query token, index into scheduled_requests
        key_positions = []  # per read/key entry, logical position within its own request
        key_group = []       # per read/key entry, index into scheduled_requests

        for group_idx, req in enumerate(scheduler_output.scheduled_requests):
            start = len(toks)
            if req.phase == RequestPhase.NEEDS_PREFILL:
                toks += req.prompt_token_ids
                write_positions = range(req.prompt_len)
                read_positions = range(req.prompt_len)
            else:  # NEEDS_DECODE
                toks.append(req.output_token_ids[-1])
                write_positions = [req.total_len - 1]
                read_positions = range(req.total_len)

            for logical_pos in write_positions:
                write_idxes.append(self._flat_slot(req.block_table, logical_pos))
                position_ids.append(logical_pos)
                query_group.append(group_idx)
            for logical_pos in read_positions:
                read_idxes.append(self._flat_slot(req.block_table, logical_pos))
                key_positions.append(logical_pos)
                key_group.append(group_idx)
            req_spans.append((req, start, len(toks)))

        position_ids_t = torch.tensor(position_ids, device=self.device)
        query_group_t = torch.tensor(query_group, device=self.device)
        key_positions_t = torch.tensor(key_positions, device=self.device)
        key_group_t = torch.tensor(key_group, device=self.device)

        # allowed[i, j]: can query token i attend to read/key entry j?
        # -- same request, and the key's position doesn't come after the query's own position.
        same_request = query_group_t[:, None] == key_group_t[None, :]
        causal = key_positions_t[None, :] <= position_ids_t[:, None]
        allowed = same_request & causal

        attention_mask = torch.zeros(len(toks), len(read_idxes), dtype=self.dtype, device=self.device)
        attention_mask.masked_fill_(~allowed, torch.finfo(self.dtype).min)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total_query, total_read]

        position_ids_for_model = position_ids_t.unsqueeze(0)  # [1, total_query]

        write_idxes_t = torch.tensor(write_idxes, device=self.device)
        read_idxes_t = torch.tensor(read_idxes, device=self.device)
        input_ids = torch.tensor(toks, device=self.device).unsqueeze(0)  # [1, total_query]

        cache = _PagedKVCache(self.kv_cache)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_for_model,
            past_key_values=None,
            cache=cache,
            read_index=read_idxes_t,
            write_index=write_idxes_t,
            use_cache=False,
        )

        last_token_logits = {req.request_id: outputs.logits[0, end - 1] for req, _, end in req_spans}
        return torch.stack([last_token_logits[req.request_id] for req in scheduler_output.scheduled_requests])
