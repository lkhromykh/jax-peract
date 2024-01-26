import tree
import numpy as np
from transformers import CLIPTokenizer

NLGoal = dict[str, np.ndarray]


class TextTokenizer:

    def __init__(self, max_length: int = 77) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.max_length = max_length

    def encode(self, text_: str) -> NLGoal:
        out = self.tokenizer(
            text_,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        return tree.map_structure(np.squeeze, out)

    def decode(self, goal: NLGoal) -> str:
        return self.tokenizer.decode(goal['input_ids'], skip_special_tokens=True)
