import os
import re
from typing import Callable, List, Optional, Tuple, Set
import csv
import time
import warnings

import os, certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Suppress XLA/JAX warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF/XLA warnings
warnings.filterwarnings('ignore', message='.*SoL config.*')


import jax
from gemma import gm
from jax import numpy as jnp
from gemma import peft

class LLMManager:
    """Manages the Gemma models (fast and pro) and the sampler."""
    def __init__(
        self,
        model_sizes: List[str] = ["1b", "4b", "12b"],
        cache_length: int = 4096,
        backend: str = "iree",
    ):
        self.models = []
        self.params = []
        self.samplers = []
        self.model_map = {size: i for i, size in enumerate(model_sizes)}
        self.tokenizer = gm.text.Gemma3Tokenizer()
        self.cache_length = int(cache_length)
        backend_norm = backend.lower().strip()
        if backend_norm == "jax":
            backend_norm = "xla"
        if backend_norm not in {"iree", "xla"}:
            raise ValueError(f"Unsupported backend: {backend}. Use 'iree' or 'xla'.")
        self.backend = backend_norm
        if self.cache_length <= 0:
            raise ValueError(f"cache_length must be > 0, got {cache_length}")
        print(self.model_map)
        self._load_models(model_sizes)

    def _load_models(self, model_sizes: List[str], load_param: bool = True):
        """Loads the specified Gemma models and parameters."""
        for size in model_sizes:
            params = None
            if size == "270m":
                print("Loading Gemma 270M model")
                model = gm.nn.Gemma3_270M(dtype=jnp.bfloat16)
                if load_param:
                    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)
            elif size == "1b":
                print("Loading Gemma 1B model")
                model = gm.nn.Gemma3_1B(dtype=jnp.bfloat16)
                if load_param:
                    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
            elif size == "4b":
                print("Loading Gemma 4B model ")
                model = gm.nn.Gemma3_4B(dtype=jnp.bfloat16)
                if load_param:
                    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
            # elif size == "4b-q":
            #     print("Load INT8 Gemma 4B model")
            #     model = gm.nn.IntWrapper(model = gm.nn.Gemma3_4B(), dtype = jnp.int8)
                
            else: raise ValueError(f"Unsupported model size: {size}")
            if params is None:
                raise ValueError("Params must be loaded for IREE runner.")
            self.models.append(model)
            self.params.append(params)
            print('!' * 40)
            sampler_cls = gm.iree.text.ChatSampler if self.backend == "iree" else gm.text.ChatSampler
            sampler = sampler_cls(
                model=model,
                params=params,
                cache_length=self.cache_length,
                multi_turn=False,
            )
            # Keep prefill attention_mask width consistent with compiled MLIR.
            if hasattr(sampler, "sampler") and hasattr(sampler.sampler, "pad_length"):
                object.__setattr__(sampler.sampler, "pad_length", (self.cache_length,))
            self.samplers.append(sampler)
        print("All models loaded successfully.")
  

    def generate(self, prompt: str, model_size: str) -> str:
        """Generates a response from a specific LLM."""
        model_idx = self.model_map.get(model_size)
        if model_idx is None: raise ValueError(f"Model size '{model_size}' not loaded.")
        print(f"using id={model_idx} model size")
        reply = self.samplers[model_idx].chat(prompt, multi_turn=False, print_stream=False, max_new_tokens=200)
        if self.backend == "iree":
            gm.iree.show_stats()
        return reply
    
if __name__ == "__main__":
    llm = LLMManager(["1b"])
    prompt = """
    You have access to the following tools:
    - kmeans: Find K centriods of N nodes (assuming tool already knows input and output). Usage kmeans(-k, <number of clusers>, -n, <number of nodes>)
    - lavaMD: find molecule movement of N random particles (assuming tool already knows input and output). Usage lavaMD(-boxes1d, <number of particles>)
    - bm25: search engine for wikipedia pages, tool for answering simple questions (assuming tool already knows input and output). Usage bm25()
    - kb: knowledge base of wikipedia pages, tool for answering hard questions (assuming tool already knows input and output). Usage kb()

    Your task is to answer the user's question. First, decide if a tool needs to be invoked. A tool needs to be invoked only if its output can help answer the question.
    If so, respond with the command to call the tool and a score out of 5 for necessity of using a tool in the following format:

    command = ....
    necessity = ....

    Question: I have 1000000 data points that I need to group into 10 distinct clusters. Can you compute all 10 centroids?
    """
    print(llm.generate(prompt, "1b"))
