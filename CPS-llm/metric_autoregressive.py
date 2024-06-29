import os
import csv
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelWithLMHead # TODO other models
from tqdm import tqdm


def get_log_prob_unigram_autoregressive(prev_token_ids, full_token_ids, tgt_idx, lm):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    print("---------------get_log_prob_unigram_autoregressive starts-----------")

    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    uncased = lm["uncased"]

    # get model hidden states
    output = model(prev_token_ids)
    hidden_states = output[0].squeeze(0)
    print("hidden_states : ", hidden_states)
    print("hidden_states size : ", hidden_states.size())
    
    hs = hidden_states[-1] # use logits for next word prediction
    print("hs : ", hs)
    print("hs size : ", hs.size())


    target_id = full_token_ids[0][tgt_idx]
    print("tgt_idx : ", tgt_idx)
    print("target_id : ", target_id)

    log_probs = log_softmax(hs)[target_id]
    print("log_probs : ", log_probs)
    print("log_probs size : ", log_probs.size())
   
    return log_probs

def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """
    print("-------get_span-------------")

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    print("seq1 : ", seq1)
    print("seq2 : ", seq2)


    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    print("template1 : ", template1)
    print("template2 : ", template2)

    return template1, template2


def mask_unigram(data, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    print("---------------mask_unigram-----------------")

    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    sent1, sent2 = data["sent_more"], data["sent_less"]  

    if uncased:
        print("------------uncased-------------")
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    print("sent1 : ", sent1)

    print("sent2 : ", sent2) 

    # tokenize
    if mask_token:
        print("--------mask_token = true-----------")
        sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
        sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

        print("sent1_token_ids : ", sent1_token_ids)
        print("sent2_token_ids : ", sent2_token_ids)  

    else:
        # append BOS token for conditional generation
        print("--------mask_token = false-----------")

        sent1_token_ids = tokenizer.encode(tokenizer.bos_token + sent1, return_tensors='pt', add_special_tokens=False)
        sent2_token_ids = tokenizer.encode(tokenizer.bos_token + sent2, return_tensors='pt', add_special_tokens=False)
        
        print("sent1_token_ids : ", sent1_token_ids)
        print("sent1_token_ids size : ", sent1_token_ids.size())

        print("sent2_token_ids : ", sent2_token_ids) 
        print("sent2_token_ids size : ", sent2_token_ids.size()) 

    # get spans of non-changing tokens
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)

    N = len(template1)  # num. of tokens that can be masked

    print("N : ", N)

    sent1_log_probs = 0.
    sent2_log_probs = 0.
    total_masked_tokens = 0
    
    
    # pass to model one word at a time for autogressive models
    # start at 1 because BOS token is prepended

    print("--------for loop starts-----------")

    for i in range(1, N):

        print("i : ", i)

        sent1_masked_token_ids = sent1_token_ids.clone().detach()[:, :template1[i]]
        sent2_masked_token_ids = sent2_token_ids.clone().detach()[:, :template1[i]]

        print("sent1_masked_token_ids : ", sent1_masked_token_ids)
        print("sent1_masked_token_ids size : ", sent1_masked_token_ids.size())

        print("sent2_masked_token_ids : ", sent2_masked_token_ids)
        print("sent2_masked_token_ids size : ", sent2_masked_token_ids.size())

        total_masked_tokens += 1

        print("total_masked_tokens : ", total_masked_tokens)
        
        score1 = get_log_prob_unigram_autoregressive(sent1_masked_token_ids, sent1_token_ids, template1[i], lm)
        score2 = get_log_prob_unigram_autoregressive(sent2_masked_token_ids, sent2_token_ids, template2[i], lm)
 
        # score1 = sentence_prob_mean(sent1_masked_token_ids, lm)
        # score2 = sentence_prob_mean(sent2_masked_token_ids, lm)

        print("score1 : ", score1)
        # print("score1 size : ", score1.size())

        print("score2 : ", score2)
        # print("score2 size : ", score2.size())

        sent1_log_probs += score1.item()
        sent2_log_probs += score2.item()

        print("sent1_log_probs : ", sent1_log_probs)
        print("sent2_log_probs : ", sent2_log_probs)
    score = {}
    
    score["sent1_score"] = sent1_log_probs
    score["sent2_score"] = sent2_log_probs

    print("score : ", score)

    return score


def evaluate(args):
    """
    Evaluate a masked language model using CrowS-Pairs dataset.
    """

    print("Evaluating:")
    print("Input:", args.input_file)
    print("Model:", args.lm_model_path)
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    # load data into panda DataFrame

    df_data = pd.read_csv(args.input_file)

    def add_full_stop(sentence):
        return sentence if sentence.endswith('.') else sentence + '.'

    df_data['sent_more'] = df_data['sent_more'].apply(add_full_stop)
    df_data['sent_less'] = df_data['sent_less'].apply(add_full_stop)

    # df_data = df_data.drop(['del1','del2','del3','del4','del5','annotations','Unnamed: 0'], axis =1)

    # columns: Gender_ID_x, Gender_ID_y, sent_x, sent_y
    # x is always queer, y is always straight
    # i.e. sent_x is "more stereotypical" and sent_y is "less stereotypical"

    # fairly hacky handling of filenames - could fix by reading config file instead of hard coding for my file structure
    # deal with trailing slash
    
    # if args.lm_model_path[-1] == '/': args.lm_model_path = args.lm_model_path[:-1] 
    # base_model_path = "../new_finetune/pretrained/" + args.lm_model_path.split('/')[-1].split("-finetuned")[0]
    # tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # if hasattr(tokenizer, 'do_lower_case'):
    #     uncased = tokenizer.do_lower_case
    # else:
    #     uncased = False
    # if "opt" in args.lm_model_path:
    #     model = AutoModelForCausalLM.from_pretrained(args.lm_model_path)
    # elif "gpt2" in args.lm_model_path or "bloom" in args.lm_model_path or "bart" in args.lm_model_path:
    #     model = AutoModelWithLMHead.from_pretrained(args.lm_model_path)
    # else:
    #     model = AutoModelForMaskedLM.from_pretrained(args.lm_model_path)   
    #  
    model = AutoModelWithLMHead.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if hasattr(tokenizer, 'do_lower_case'):
        uncased = tokenizer.do_lower_case
    else:
        uncased = False

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    mask_token = tokenizer.mask_token

    print("mask_token : ", mask_token)

    log_softmax = torch.nn.LogSoftmax(dim=0)
    vocab = tokenizer.get_vocab()

    # print("vocab : ", vocab)

    with open(args.lm_model_path + ".vocab", "w") as f:
        f.write(json.dumps(vocab))

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
    }

    # score each sentence. 
    # each row in the dataframe has the sentid and score for pro and anti stereo.
    df_score = pd.DataFrame(columns=['sent_more', 'sent_less', 
                                     'sent_more_score', 'sent_less_score',
                                     'score', 'bias_target_group'])

    print("df_score : ", df_score)

    total_pairs = 0
    stereo_score =  0

    # dict for keeping track of scores by category
    category_scores = {group: {'count': 0, 'score': 0, 'metric': None} for group in df_data.Gender_ID_x.unique()}

    N = 0
    neutral = 0
    total = len(df_data.index)

    print("total : ", total)

    print("----------for loop starts ------------")
    with tqdm(total=total) as pbar:
        for index, data in df_data.iterrows():
            bias = data['Gender_ID_x']
            score = mask_unigram(data, lm)

            print("score : ", score)

            # round all scores to 3 places
            for stype in score.keys():
                score[stype] = round(score[stype], 3)
                print("score[stype] : ", score[stype])

            N += 1
            print("N : ", N)

            category_scores[bias]['count'] += 1
            pair_score = 0
            pbar.update(1)
            if score['sent1_score'] == score['sent2_score']:
                neutral += 1
            else:
                total_pairs += 1
                if score['sent1_score'] > score['sent2_score']:
                        stereo_score += 1
                        category_scores[bias]['score'] += 1
                        pair_score = 1
                        
            sent_more = data['sent_more']
            sent_less = data['sent_less']
            sent_more_score = score['sent1_score']
            sent_less_score = score['sent2_score']

            df_score = df_score.append({'sent_more': sent_more,
                                        'sent_less': sent_less,
                                        'sent_more_score': sent_more_score,
                                        'sent_less_score': sent_less_score,
                                        'score': pair_score,
                                        'bias_target_group': bias
                                      }, ignore_index=True)


    df_score.to_csv(args.output_file)
    if args.summary_file:
        summary_path = args.summary_file
    else:
        summary_path = args.output_file + ".summary"

    
    with open(summary_path, 'w') as f:
        f.write('Total examples: ' + str(N) + '\n')
        f.write("Num. neutral:" + str(neutral) + ", % neutral: " + str(round(neutral / N * 100, 2)) + '\n')
        f.write('Test data Overall Score: ' + str(round(stereo_score / N * 100, 2)) + '\n')
        f.write('Score Breakdown by Target of Bias:\n')
        for k, v in category_scores.items():
            f.write("Category: " + k + '\n')
            f.write("    Number of examples: " + str(v['count']) + '\n')
            if v['count'] > 0:
                v['metric'] = round(v['score'] / v['count'] * 100, 2)
                f.write("    Bias score against group " + k + ": " + str(v['metric']) + '\n')

        f.write("For pasting into spreadsheet (Order Overall, 'Male'):")
        # use list of keys instead of category_scores.items() to force order to match the spreadsheet
        f.write(str(round(stereo_score / N * 100, 2)) + ", " + ", ".join([str(category_scores[key]['metric']) for key in ['Male']]))

    print('=' * 100)
    print("Output written to: " + args.output_file)
    print("summary stats written to: " + summary_path)

    print("For pasting into spreadsheet (Order Overall, 'Male'):\n")
    # use list of keys instead of category_scores.items() to force order to match the spreadsheet
    print(str(round(stereo_score / N * 100, 2)) + ", " + ", ".join([str(category_scores[key]['metric']) for key in ['Male']]) + "\n")

    print('=' * 100)


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="path to input file")
parser.add_argument("--lm_model_path", type=str, help="path to pretrained LM model to use")
parser.add_argument("--output_file", type=str, help="path to output file with sentence scores")
parser.add_argument("--summary_file", type=str, help="path to output summary stats", required=False)

args = parser.parse_args()
evaluate(args)

# python metric_autoregressive.py --input_file data\test_data_ag.csv --lm_model_path gpt2  --output_file output_autoreg 
