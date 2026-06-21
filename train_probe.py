import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier probe for PROBIX.")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", default="sst2")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_examples", type=int, default=2000)
    return parser.parse_args()


args = parse_args()

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.dataset_utils import load_dataset_custom

model_name = args.model_name
dataset_name = args.dataset_name
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

class DatasetArgs:
    def __init__(self, dataset_name, num_examples):
        self.dataset_name = dataset_name
        self.num_examples = num_examples

start_time = time.time()
dataset_args = DatasetArgs(dataset_name=dataset_name, num_examples=args.num_examples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device) 
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

model.eval()

num_labels = 4 if dataset_name == "AG-News" else 2
classifier_probe = nn.Linear(model.config.hidden_size, num_labels).to(device)
classifier_probe.weight.data.normal_(mean=0.0, std=0.02) 
classifier_probe.bias.data.zero_()
optimizer = AdamW(classifier_probe.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

dataset = load_dataset_custom(dataset_args)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Training classifier probe for {dataset_name}...")
for epoch in range(num_epochs):
    classifier_probe.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}")
        texts, labels = batch
        labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device, dtype=torch.long)
        print(f"Labels values: {labels}") 

        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, -1, :].float()
            print(f"Hidden states min/max: {hidden_states.min()}, {hidden_states.max()}") 
        
        hidden_states = torch.clamp(hidden_states, min=-1e9, max=1e9)
        
        logits = classifier_probe(hidden_states)
        print(f"Logits min/max: {logits.min()}, {logits.max()}") 
        loss = loss_fn(logits, labels)
        print(f"Batch loss: {loss.item()}") 
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(classifier_probe.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

end_time = time.time()
execution_time = end_time - start_time
save_path = f"classifier_probe_{model_name.replace('/', '_')}_{dataset_name}.pt"
torch.save(classifier_probe.state_dict(), save_path)
print(f"Classifier probe saved to {save_path}")
print(f"Execution Time: {execution_time:.2f} seconds")