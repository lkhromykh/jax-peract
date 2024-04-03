from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from dm_env import specs

import transformers
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from transformers.tokenization_utils_base import BatchEncoding
transformers.logging.set_verbosity_error()

Array: TypeAlias = np.ndarray
Tokens: TypeAlias = BatchEncoding[str, Array]


class CLIP:

    def __init__(self,
                 max_length: int = 77,
                 pretrained_model_name_or_path: str = 'openai/clip-vit-base-patch32'  # 'openai/clip-vit-large-patch14'
                 ) -> None:
        self._tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path)
        self._model = FlaxCLIPTextModel.from_pretrained(pretrained_model_name_or_path, dtype=jnp.bfloat16)
        self.max_length = min(max_length, self._model.config.max_position_embeddings)
        self._cache = (None, None)

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

    def encode(self, input_: str | Array) -> Array:
        if isinstance(input_, Array):
            input_ = input_.item()
        if not isinstance(input_, str):
            raise TypeError(f'Wrong argument type: {input_}.')
        prev_hash, prev_emb = self._cache
        cur_hash = hash(input_)
        if cur_hash == prev_hash:
            return prev_emb
        tokens = self.tokenize(input_)
        tokens = jax.device_put(dict(tokens))
        emb = jax.jit(self._model)(**tokens)
        emb = emb.last_hidden_state.astype(jnp.float32).squeeze()
        self._cache = (cur_hash, emb)
        return emb

    def observation_spec(self) -> specs.Array:
        hidden_size = self._model.config.hidden_size
        return specs.Array((self.max_length, hidden_size), dtype=np.float32, name='text_emb')
