from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = '/home/anjana/Project/LLM/llama-2-7b-chat.ggmlv3.q8_0.bin'

model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

def summarize_content(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def chatbot():
    print("Hello! I can summarize any content for you.")
    while True:
        user_input = input("Enter text to summarize (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        summary = summarize_content(user_input)
        print(f"Summary: {summary}")
