"""
English to Spanish Dataset 

Typical Usage:
`train_dataset = EnglishToSpanish('train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

batch = next(iter(train_loader))
print('X shape:', batch['input_ids'].shape)
print('y shape:', batch['labels'].shape)`
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset('Helsinki-NLP/opus_books', 'en-es')

tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')

class EnglishToSpanish(Dataset):
    def __init__(self, split, max_length=128):
        self.data = dataset[split]
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        source = sample['translation']['en']
        target = sample['translation']['es']

        source_enc = tokenizer(source, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        target_enc = tokenizer(target, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {'input_ids': source_enc['input_ids'].squeeze(0),
                'attention_mask': source_enc['attention_mask'].squeeze(0),
                'labels': target_enc['input_ids'].squeeze(0)}