import jax
import numpy as np
from dm_env import specs

import transformers
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from transformers.tokenization_utils_base import BatchEncoding
transformers.logging.set_verbosity_error()

Array = np.ndarray
Tokens = BatchEncoding[str, Array]


class CLIP:

    PRETRAINED_PATH = 'openai/clip-vit-base-patch32'

    def __init__(self, max_length: int = 77) -> None:
        self._tokenizer = CLIPTokenizer.from_pretrained(CLIP.PRETRAINED_PATH)
        self._model = FlaxCLIPTextModel.from_pretrained(CLIP.PRETRAINED_PATH)
        self.max_length = max_length
        self._cache = (0, None)

    def tokenize(self, text_: str) -> Tokens:
        return self._tokenizer(
            text_,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )

    def detokenize(self, tokens: Tokens) -> str:
        return self._tokenizer.batch_decode(tokens['input_ids'], skip_special_tokens=True)

    def encode(self, input_: str | np.ndarray) -> Array:
        if isinstance(input_, np.ndarray):
            input_ = input_.item()
        if not isinstance(input_, str):
            raise ValueError(f'Wrong argument type: {input_}.')
        prev_hash, prev_emb = self._cache
        cur_hash = hash(input_)
        if cur_hash == prev_hash:
            return prev_emb
        tokens = self.tokenize(input_)
        emb = jax.jit(self._model)(**tokens).last_hidden_state
        emb = jax.device_get(emb).squeeze()
        self._cache = (cur_hash, emb)
        return emb

    def observation_spec(self) -> specs.Array:
        return specs.Array((self.max_length, 512), dtype=np.float32, name='text_emb')
