import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def forward(self, scheduler_output) -> torch.Tensor:
        """Batched forward pass over a SchedulerOutput's requests."""
        raise NotImplementedError  # TODO
