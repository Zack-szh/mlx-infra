from pathlib import Path
from mlx_lm import generate
from mlx_lm.utils import load_model

from .tokenizer import load_tokenizer

class Engine: 
    def __init__(self, model_path: str): 
        # load_model() returns (nn.Module, config_dict)
        self.model, _config = load_model(Path(model_path))
        self.tokenizer = load_tokenizer(model_path)

    def generate(self, prompt: str, max_tokens: int = 512) -> str: 
        # use mlx_lm.generate for now as wrapper
        # later replace step by step
        return generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            verbose=True,
        )
