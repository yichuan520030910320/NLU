import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

class NLIDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} 
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item



    def __len__(self):
        return len(self.encodings.input_ids)

def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()

    sentences = [(json.loads(line)['text_a'], json.loads(line)['text_b']) for line in lines]
    labels = [json.loads(line)['label'] for line in lines] if 'label' in json.loads(lines[0]) else None

    return sentences, labels

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Initialize the model and move to GPU
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  
model.to('cuda')   

# Load the data
train_path = '/home/ycwang/NLI/dataset/train.jsonl'
test_path = '/home/ycwang/NLI/dataset/test.jsonl'
train_sentences, train_labels = load_data(train_path)
test_sentences, test_labels = load_data(test_path) 



# Convert labels to integers
label_mapping = {"entailment": 0, "contradiction": 1, "neutral": 2}
train_labels = [label_mapping[label] for label in train_labels]
# test_labels = [label_mapping[label] for label in test_labels]

# Tokenize the dataset, this will return a dictionary with the keys input_ids, token_type_ids and attention_mask
train_encodings = tokenizer(train_sentences, truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(test_sentences, truncation=True, padding=True, return_tensors='pt')

# Create Dataset objects
train_dataset = NLIDataset(train_encodings, train_labels)
test_dataset = NLIDataset(test_encodings, None)  # no labels for the test dataset
# Define the training arguments  
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=1,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',    
    gradient_accumulation_steps=2,  # Added gradient accumulation  
    fp16=True,
)  


# Create a Trainer instance
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
)

# Train the model
trainer.train()

print('finished training')
# Make predictions on test set
predictions = trainer.predict(test_dataset).predictions.argmax(-1)

# Convert numeric predictions to labels
label_mapping = {0: "entailment", 1: "contradiction", 2: "neutral"}
predictions = [label_mapping[pred] for pred in predictions]

# Write predictions to a CSV file
output_file = '/home/ycwang/NLI/predictions.csv'
df = pd.DataFrame({"ID": range(len(predictions)), "Label": predictions})
df.to_csv(output_file, index=False)
