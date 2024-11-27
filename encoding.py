import torch
import pickle
import pandas as pd
import multiprocessing

from tqdm import tqdm
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForTextEncoding

@torch.no_grad()
def encode(s: List[str]) -> torch.Tensor:
    tokenized_s = tokenizer(s, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized_s.input_ids
    attention_mask = tokenized_s.attention_mask

    # batches to prevent memory overflow
    hidden_state = model(input_ids).last_hidden_state
    lens = attention_mask.sum(dim=1)
    features = (hidden_state * attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
    return features

def encode_premises(premises: List[str], filename: str, batch_size: int = 8):
    premise_encs = []
    print(f"Encoding premises in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(premises), batch_size)):
        batch = premises[i:i + batch_size]
        enc = encode(batch)
        premise_encs.append(enc.cpu())

    premise_encs = torch.cat(premise_encs, dim=0)
    premise_encs = premise_encs.numpy()
    data = {'premises': premises, 'encodings': premise_encs}

    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Premises and encodings saved to {filename}")

file_path = 'data/dependencies_mathlib_v4.14.0-rc1.jsonl'
df = pd.read_json(file_path, lines=True)
all_premises = df['full_decl_no_comments'].tolist()

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
model = AutoModelForTextEncoding.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
model.eval()

# use all available CPU cores
num_cpu_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cpu_cores)
print(f"Using {num_cpu_cores} CPU cores for processing.")

encode_premises(all_premises, 'premises.pkl', batch_size=32)