
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import numpy as np
import torch

class CodeGen:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True, attn_impl='torch')
        self.model.to(device='cuda:0', dtype=torch.bfloat16)

    def generate(self, data):
        prompt = data['prompt']
        max_tokens = data.get('max_tokens', 16)

        temperature = data.get('temperature', 0.2)
        if temperature == 0.0:
            temperature = 1.0
            top_k = 1
        else:
            top_k = data.get('top_k', 0)

        top_p = data.get('top_p', 1.0)
        frequency_penalty = data.get('frequency_penalty', 1.0)

        n = data.get('n', 1)
        num_beams = 6
        num_beam_groups = 3
        if n > 6:
            if n % 2 == 0:
                num_beams = n
                num_beam_groups = n/2
            else:
                num_beams = n+1
                num_beam_groups = (n+1)/2

        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(device='cuda:0') # type: ignore

        outputs = self.model.generate(inputs,
                                      max_length=max_tokens,
                                      num_return_sequence=n,
                                      repetition_penalty=frequency_penalty,
                                      temperature=temperature,
                                      top_p=top_p,
                                      top_k=top_k,
                                      num_beams=num_beams,
                                      num_beam_groups=num_beam_groups,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                    )

        prompt_len = len(prompt)
        decoded = self.tokenizer.batch_decode([out[prompt_len:prompt_len+g] for g, out in outputs])
