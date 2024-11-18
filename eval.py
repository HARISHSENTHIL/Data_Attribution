import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_name = './fine_tuned_model'  # Path to the saved fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model.to(device)

def evaluate_model(question):
    input_text = f"Q: {question} A:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)  
    
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<id>" in response:
        answer, doc_id = response.split("<id>")
        return answer.strip(), doc_id.strip()
    
    return response.strip(), None

question_to_evaluate = "Who invented laws of motion?"
answer, doc_id = evaluate_model(question_to_evaluate)
print(f"Answer: {answer}, Document ID: {doc_id}") 
