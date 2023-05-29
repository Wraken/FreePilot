from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import time

# load model
model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b',
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=True,
                                            ).to(device='cuda:0', dtype=torch.bfloat16)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)

# start_time = time.time()

# x = tokenizer.encode('def fibonacci(n): ', return_tensors='pt')
# x = x.to(device='cuda:0') # type: ignore
# y = model.generate(x, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

# generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print("py")
# print("--- %s seconds ---" % (time.time() - start_time))
# print(generated_code)

# start_time = time.time()

# x = tokenizer.encode('package main', return_tensors='pt')
# x = x.to(device='cuda:0')  # type: ignore
# y = model.generate(x, max_length=100, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

# generated_code = tokenizer.decode(y[0], clean_up_tokenization_spaces=False)
# print("go")
# print("--- %s seconds ---" % (time.time() - start_time))
# print(generated_code)

prompt = 'func main() {'

start_time = time.time()

inputs = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda:0')
print(inputs, inputs.size(), inputs.size(dim=1))

outputs = model.generate(inputs, do_sample=False, num_beams=6, num_beam_groups=2,max_length=100, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=4, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

prompt_len = len(prompt)
decoded = tokenizer.batch_decode(outputs)

print("go")
print("--- %s seconds ---" % (time.time() - start_time))
for i, d in enumerate(decoded):
    print(i, d)