import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_text(text):
    return tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=100
    )["input_ids"].to(device)
