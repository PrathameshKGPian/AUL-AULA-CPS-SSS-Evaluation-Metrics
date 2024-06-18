tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')


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

    # Calculate the softmax probabilities
    probs = softmax(shift_logits, dim=-1)

    # Gather the probabilities of the actual token IDs
    gathered_probs = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Compute the mean probability across the tokens
    mean_prob = torch.mean(gathered_probs).item()

    return mean_prob
