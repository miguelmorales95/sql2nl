import os
from typing import Optional
from .prompts import INSTRUCTION_TEMPLATE

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except Exception as e:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    torch = None

class Seq2SeqExplainer:
    def __init__(self, model_dir: Optional[str] = None, base_model: str = "t5-small", device: Optional[str] = None):
        if AutoTokenizer is None:
            raise ImportError("Transformers not installed. Please `pip install transformers torch accelerate`.")
        self.model_name = model_dir or base_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, sql: str, max_new_tokens: int = 96) -> str:
        prompt = INSTRUCTION_TEMPLATE.format(sql=sql)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
