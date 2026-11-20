"""Local Hugging Face LLM. CPU by default; 4-bit Qwen3-8B if CUDA GPU exists."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

_tok = _model = None


def load():
    global _tok, _model
    if _model is None:
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16)
            _tok = AutoTokenizer.from_pretrained(config.LLM_MODEL_GPU)
            _model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_GPU, quantization_config=bnb, device_map="auto")
        else:
            _tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
            _model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL,
                dtype=torch.bfloat16,        # ~6 GB RAM for the 3B model
                low_cpu_mem_usage=True)
        _model.eval()
    return _tok, _model


def chat(system, user, max_new_tokens=800):
    tok, model = load()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()