import tiktoken
from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Separate input and target sequences
    input_ids, target_ids = zip(*batch)
    
    # Pad sequences to the length of the longest sequence in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    return input_ids_padded, target_ids_padded


class TiktokenTokenizer:
    def __init__(self):
        self.tiktoken_tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tiktoken_tokenizer.encode(text)

    def decode(self, indices):
        return self.tiktoken_tokenizer.decode(indices)


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        if len(token_ids) >= max_length:
            for i in range(0, len(token_ids) - max_length + 1, stride):
                self.input_ids.append(token_ids[i:i + max_length])
                self.target_ids.append(token_ids[i + stride:i + max_length + stride])
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.target_ids[idx])


class DataLoaderFactory:
    def __init__(self, tokenizer_name="gpt2"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def create_dataloader_v1(self, txt, batch_size=4, max_length=256,
                             stride=128, shuffle=True, drop_last=True,
                             num_workers=0):
        dataset = GPTDatasetV1(txt, self.tokenizer, max_length, stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        return dataloader