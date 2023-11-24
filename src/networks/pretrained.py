import jax
import numpy as np
from transformers import CLIPTokenizer, FlaxCLIPTextModel


class TextEncoder:

    def __init__(self, max_length):
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.model = FlaxCLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')

    def __call__(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text,
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='np')
        output = self.model(**inputs)
        output = jax.numpy.squeeze(output.last_hidden_state, 0)
        return jax.device_get(output)
