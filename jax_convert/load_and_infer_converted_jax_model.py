from transformers import AutoTokenizer, GPTJForCausalLM, GPTJConfig, GPT2Tokenizer
import torch

def main():
    # TODO: use "cuda" --> warning: cuda version not match with 3090ti --> need to reinstall nvidia toolkit
    device = "cpu"
    config = GPTJConfig.from_json_file("config_converted_jax_model.json")
    model = GPTJForCausalLM(config).to(device)

    # TODO: use GPT-j tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# This prompts only works on fine-tuned models. Use just natural langauge prompt for not fine-tuned model.
# TODO: add cutting context by max length
#    def get_prompt(context, proof_state):
#      return f" Oxford {context}" + f"<ISA_OBS>{proof_state} Cambridge"

#    proof_state = "proof (prove)\ngoal (1 subgoal):\n 1. size (del_min t) = size t - 1"
#    context = "lemma size_del_min: assumes \"braun t\" shows \"size(del_min t) = size t - 1\""
#    prompt = get_prompt(proof_state, context)

    prompt = "My name is"
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids

    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)

    print(tokenizer.batch_decode(gen_tokens)[0])

if __name__ == "__main__":
    main()

