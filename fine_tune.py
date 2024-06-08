import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader

squad_dataset = load_dataset("squad")

# format dataset
def preprocess_data(batch):
    input_texts = [f"generate question: {context}" for context in batch['context']]
    target_texts = batch['question']
    return {"input_text": input_texts, "target_text": target_texts}

train_split = squad_dataset['train'].map(preprocess_data, batched = True)
validation_split = squad_dataset['validation'].map(preprocess_data, batched = True)

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# convert dataset into tokens
def tokenize_dataset(batch):
    inputs = tokenizer(batch['input_text'], padding = 'max_length', truncation = True, max_length = 512)
    targets = tokenizer(batch['target_text'], padding = 'max_length', truncation = True, max_length = 32)
    
    inputs['labels'] = targets['input_ids']
    return inputs

train_split = train_split.map(tokenize_dataset, batched = True)
validation_split = validation_split.map(tokenize_dataset, batched = True)

train_split.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
validation_split.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])

train_loader = DataLoader(train_split, batch_size = 8, shuffle = True)
validation_loader = DataLoader(validation_split, batch_size = 8)

model = T5ForConditionalGeneration.from_pretrained('t5-small')
optimizer = AdamW(model.parameters(), lr = 5e-5)
epoch_num = 3
num_training_steps = len(train_loader) * epoch_num
lr_scheduler = get_scheduler(name = "linear", optimizer = optimizer, num_warmup_steps = 0, num_training_steps = num_training_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
logging_steps = 100
for epoch in range(3):
    total_loss = 0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss.item()
        
        if (step + 1) % logging_steps == 0:
            print(f"Epoch: {epoch + 1}, Step: {step + 1}, Loss: {loss.item():.4f}, Avg Loss: {total_loss / (step + 1):.4f}")

# save model and tokenizer
save_directory = "fine-tuned"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
