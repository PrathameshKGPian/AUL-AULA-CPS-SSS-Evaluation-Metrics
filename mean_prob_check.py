import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.functional import softmax
import numpy as np, pandas as pd
from tqdm import tqdm

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

df = pd.read_csv("Data_for_CPS_LLM.csv")

total = len(df.index)

def sentence_prob_mean(text):
    # Tokenize the input text and add special tokens
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # model outputs 
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # logits are the model outputs before applying softmax

    # Shift logits and labels so that tokens are aligned:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Softmax probabilities
    probs = softmax(shift_logits, dim=-1)

    # The probabilities of the actual token IDs
    gathered_probs = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
   
    # The mean probability across the tokens
    mean_prob = torch.mean(gathered_probs).item()

    return mean_prob

def remove_full_stop(sentence):
    return sentence[:-1] if sentence.endswith('.') else sentence

df['sent_x'] = df['sent_x'].apply(remove_full_stop)
df['sent_y'] = df['sent_y'].apply(remove_full_stop)

with tqdm(total=total) as pbar:
    for idx, row in df.iterrows():
        df.at[idx, 'sent_x_score'] = sentence_prob_mean(row['sent_x'])
        df.at[idx, 'sent_y_score'] = sentence_prob_mean(row['sent_y'])
        pbar.update(1)

df.to_csv("Out_wo_fstp.csv")
