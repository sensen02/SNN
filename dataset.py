import torch
from torch.utils.data import Dataset
import json
import os
import pickle

class CharTokenizer:
    def __init__(self, char_list):
        self.vocab = ["<pad>", "<bos>", "<eos>"] + char_list
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_id = self.char_to_id["<pad>"]
        self.bos_id = self.char_to_id["<bos>"]
        self.eos_id = self.char_to_id["<eos>"]
        
    def encode(self, text, add_bos=True, add_eos=True):
        ids = [self.char_to_id[c] for c in text if c in self.char_to_id]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join([self.id_to_char[i] for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]])

    @property
    def vocab_size(self):
        return len(self.vocab)

def get_or_create_tokenizer(corpus_path, vocab_cache_path):
    if os.path.exists(vocab_cache_path):
        try:
            with open(vocab_cache_path, "rb") as f:
                tokenizer = pickle.load(f)
            print(f"Loaded cached tokenizer with vocab size {tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            print(f"Failed to load cached tokenizer: {e}. Rebuilding...")

    print("Building tokenizer from corpus...")
    vocab = set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                for char in text:
                    vocab.add(char)
            except:
                pass
    char_list = sorted(list(vocab))
    tokenizer = CharTokenizer(char_list)
    
    os.makedirs(os.path.dirname(vocab_cache_path), exist_ok=True)
    with open(vocab_cache_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Created tokenizer with vocab size {tokenizer.vocab_size}")
    return tokenizer

class PackedChineseDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, chunk_len=512, cache_path=None):
        self.chunk_len = chunk_len
        
        if cache_path and os.path.exists(cache_path):
            try:
                print(f"Loading cached dataset from {cache_path}...")
                with open(cache_path, "rb") as f:
                    self.chunks = pickle.load(f)
                print(f"Loaded {len(self.chunks)} packed chunks.")
                return
            except Exception as e:
                print(f"Failed to load cached dataset: {e}. Rebuilding...")
            
        print("Tokenizing corpus and packing into chunks...")
        all_ids = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                        all_ids.extend(ids)
                except:
                    pass
        
        # Split into chunks of chunk_len
        num_chunks = len(all_ids) // chunk_len
        if num_chunks == 0:
            padding = [tokenizer.pad_id] * (chunk_len - len(all_ids))
            self.chunks = [torch.tensor(all_ids + padding, dtype=torch.long)]
        else:
            all_ids = all_ids[:num_chunks * chunk_len]
            self.chunks = []
            for i in range(num_chunks):
                chunk = all_ids[i * chunk_len : (i + 1) * chunk_len]
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
                
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.chunks, f)
            print(f"Saved cached dataset with {len(self.chunks)} chunks to {cache_path}")
            
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        return self.chunks[idx]
