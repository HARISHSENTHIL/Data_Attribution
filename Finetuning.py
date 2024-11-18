
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
model_name = 'gpt2'  
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForCausalLM.from_pretrained(model_name)

def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
    return data

file_path = 'synthetic_data_100.jsonl'  
train_data = load_dataset(file_path)

def format_data(entry):
    input_text = f"Q: {entry['question']} A: {entry['answer']} <id>{entry['id']}</id>"
    return {"input_text": input_text}

formatted_data = [format_data(entry) for entry in train_data]
dataset = Dataset.from_dict({"input_text": [d["input_text"] for d in formatted_data]})

def tokenize_function(examples):
    tokenized = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    num_train_epochs=20,
    logging_steps=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

trainer.save_model('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')  

# ----------------------------------------------------------------#


def format_data(entry):
    input_text = f"C: {['context']} Q: {entry['question']} A: {entry['answer']} <id>{entry['id']}</id>"
    return {"input_text": input_text} 

