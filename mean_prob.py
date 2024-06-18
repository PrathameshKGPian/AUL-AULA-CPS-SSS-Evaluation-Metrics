#!/usr/bin/env python3
import sys
import argparse
import torch
import re
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.functional import softmax
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


def sentence_prob_mean(text):
    # Tokenize the input text and add special tokens
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Obtain model outputs
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # logits are the model outputs before applying softmax

    # Shift logits and labels so that tokens are aligned:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # softmax probabilities
    probs = softmax(shift_logits, dim=-1)

    # the probabilities of the actual token IDs
    gathered_probs = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # mean probability across the tokens
    mean_prob = torch.mean(gathered_probs).item()

    return mean_prob
