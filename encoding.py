import torch
import pickle
import pandas as pd
import multiprocessing

from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForTextEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@torch.no_grad()
def encode(s: List[str]) -> torch.Tensor:
    tokenized_s = tokenizer(s, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = tokenized_s.input_ids
    attention_mask = tokenized_s.attention_mask

    # batches to prevent memory overflow
    hidden_state = model(input_ids).last_hidden_state
    lens = attention_mask.sum(dim=1)
    features = (hidden_state * attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
    return features

def encode_premises(premises: List[str], filename: str, batch_size: int = 8):
    premise_encs = []
    print(f"Encoding premises on {device} in batches of {batch_size}...")
    
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

@torch.no_grad()
def retrieve(query: str, premises_file: str, k: int) -> List[str]:
    """Retrieve the top-k premises given a query."""
    with open(premises_file, 'rb') as f:
        data = pickle.load(f)

    premises = data['premises']
    encodings = torch.tensor(data['encodings'])
    query_enc = encode(query)

    scores = (query_enc @ encodings.T)
    topk = scores.topk(k).indices.tolist()
    return [premises[i] for i in topk]

def main():
    file_path = 'data/dependencies_mathlib_v4.14.0-rc1.jsonl'
    output_file = 'premises.pkl'
    batch_size = 32 # adjust to not crash

    print(f"Loading data from {file_path}...")
    df = pd.read_json(file_path, lines=True)
    all_premises = df['full_decl_no_comments'].tolist()

    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
    model = AutoModelForTextEncoding.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
    model.eval()
    model.to(device)

    encode_premises(all_premises, output_file, batch_size=batch_size)

if __name__ == "__main__":
    main()