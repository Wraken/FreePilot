
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed
import numpy as np
import torch
import time
import random
import string
import json

class CodeGen:
    def __init__(self):
        config = AutoConfig.from_pretrained(
            "replit/replit-code-v1-3b",
            trust_remote_code=True
        )
        config.attn_config['attn_impl'] = 'triton'
        self.tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b',
                                            config=config,
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=True,
                                            ).to(device='cuda:0', dtype=torch.bfloat16)

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
        prompt_token_len = inputs.size(dim=1)

        print("generate:", prompt, prompt_token_len, max_tokens)

        outputs = self.model.generate(inputs,
                                      do_sample=True,
                                      max_new_tokens=max_tokens,
                                      num_return_sequences=n,
                                      repetition_penalty=frequency_penalty,
                                      temperature=temperature,
                                      top_p=top_p,
                                      top_k=top_k,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                    )

        output_lens = [len(o)-prompt_token_len for o in outputs]
        decoded = self.tokenizer.batch_decode([out[prompt_token_len:prompt_token_len + g] for g,out in zip(output_lens, outputs)])


        choices = []
        for i, (text, tokens, g) in enumerate(zip(decoded, outputs, output_lens)):
            reason = "length" if max_tokens == g else "stop"
            lpdict = None

            choice = {
                'text': text,
                'index': i,
                'finish_reason': reason,
                'logprobs': lpdict,
            }
            choices.append(choice)

        completion = {
            'id': None,  # fill in
            'model': 'codegen',
            'object': 'text_completion',
            'created': int(time.time()),
            'choices': None,  # fill in
            'usage': {
                'completion_tokens': int(sum(output_lens)),
                'prompt_tokens': int(prompt_token_len),
                'total_tokens': int(sum(output_lens) + prompt_token_len),
            }
        }
        return completion, choices

    @staticmethod
    def random_completion_id():
        return 'cmpl-' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(29))

    def streamed_response(self, completion, choices):
        for c in choices:
            completion['id'] = self.random_completion_id()
            completion['choices'] = [c]
            yield f'{json.dumps(completion)}'
        yield '[DONE]'

    def non_streamed_response(self, completion, choices) -> str:
        completion['id'] = self.random_completion_id()
        completion['choices'] = choices
        return json.dumps(completion)

    def __call__(self, data: dict):
        st = time.time()
        try:
            completion, choices = self.generate(data)
        except Exception as e:
            # status: unavailable -- this happens if the `model` string is invalid
            print(e)
            completion = {}
            choices = []
        ed = time.time()
        print(f"Returned completion in {(ed - st) * 1000} ms")
        if data.get('stream', False):
            return self.streamed_response(completion, choices)
        else:
            return self.non_streamed_response(completion, choices)