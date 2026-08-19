import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


class ModelRunner:
    """Wraps the reused HF model/tokenizer (Decision 3: model internals are
    not the point, only the batching/memory orchestration around them is).
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.model.eval()

    def forward(self, scheduler_output) -> torch.Tensor:
        """Batched forward pass over a SchedulerOutput's requests."""
        raise NotImplementedError  # TODO
