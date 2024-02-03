import numpy as np
from transformers import CLIPTokenizer, FlaxCLIPTextModel

Array = np.ndarray
Tokens = dict[str, Array]


class CLIP:

    PRETRAINED_PATH = 'openai/clip-vit-base-patch32'

    def __init__(self, max_length: int = 77) -> None:
        self._tokenizer = CLIPTokenizer.from_pretrained(CLIP.PRETRAINED_PATH)
        self._model = FlaxCLIPTextModel.from_pretrained(CLIP.PRETRAINED_PATH)
        self.max_length = max_length

    def tokenize(self, text_: str) -> Tokens:
        return self._tokenizer(
            text_,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )

    def detokenize(self, goal: Tokens) -> str:
        return self._tokenizer.decode(goal['input_ids'], skip_special_tokens=True)

    def encode(self, input_: str | Tokens) -> Array:
        if not isinstance(input_, dict):
            input_ = self.tokenize(input_)
        return self._model(**input_).last_hidden_state
