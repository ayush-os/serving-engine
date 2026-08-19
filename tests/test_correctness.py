"""Phase 1 correctness checkpoint: engine output must match plain HF
.generate() token-for-token on identical prompts. Both paths must use
greedy decoding (do_sample=False) -- the engine has no sampling logic
beyond argmax, so this is the only fair comparison.
"""
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from serving_engine.engine import LLMEngine
from serving_engine.model_runner import MODEL_NAME

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In machine learning, a transformer is",
]
MAX_NEW_TOKENS = 32


@pytest.fixture(scope="module")
def hf_reference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda().eval()
    return model, tokenizer


def hf_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_matches_hf_generate(hf_reference, prompt):
    model, tokenizer = hf_reference
    expected = hf_generate(model, tokenizer, prompt, MAX_NEW_TOKENS)

    engine = LLMEngine(num_gpu_blocks=1024)
    [actual] = engine.generate([prompt], max_new_tokens=MAX_NEW_TOKENS)

    assert actual == expected
