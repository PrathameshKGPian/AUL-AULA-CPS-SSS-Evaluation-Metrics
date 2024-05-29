import json
import argparse
import torch
import difflib

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        choices=['cp', 'ss'],
                        help='Path to evaluation dataset.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to result text file')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--method', type=str, required=True,
                        choices=['aula', 'aul', 'cps', 'sss'])
    args = parser.parse_args()

    return args


def load_tokenizer_and_model(args):
    '''
    Load tokenizer and model to evaluate.
    '''
    if args.model == 'bert':
        pretrained_weights = 'bert-base-cased'
    elif args.model == "roberta":
        pretrained_weights = 'roberta-large'
    elif args.model == "albert":
        pretrained_weights = 'albert-large-v2'
    else:
        pretrained_weights = args.model
    model = AutoModelForMaskedLM.from_pretrained(pretrained_weights,
                                                 output_hidden_states=True,
                                                 output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    print("\nmodel : ", model, "\ntokenzier :", tokenizer)

    return tokenizer, model


def get_span(seq1, seq2, operation):
    '''
    Extract spans that are shared or diffirent between two sequences.

    Parameters
    ----------
    operation: str
        You can select "equal" which extract spans that are shared between
        two sequences or "diff" which extract spans that are diffirent between
        two sequences.
    '''
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    print("\nseq1 : ", seq1, "\nseq2 : ", seq2)

    matcher = difflib.SequenceMatcher(None, seq1, seq2)

    print("\nmatcher : ", matcher)

    template1, template2 = [], []
    for op in matcher.get_opcodes():

        print("\nop : ", op)

        if (operation == 'equal' and op[0] == 'equal') \
                or (operation == 'diff' and op[0] != 'equal'):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    print("\ntemplate1 : ", template1, "\ntemplate2 : ", template2)
    return template1, template2


def get_rank_for_gold_token(log_probs, token_ids):
    '''
    Get rank for gold token from log probability.
    '''
    sorted_indexes = torch.sort(log_probs, dim=1, descending=True)[1]
    
# pro_token_ids :  tensor([[ 101, 1124, 1577,  112,  189, 2482, 1149, 1103, 2486, 1114, 1103, 8090,
#           117, 1133, 1119, 1793, 1136, 1106, 1243, 1315, 1205, 1113, 1471,  117,
#          1211, 1602, 4067, 1274,  112,  189, 1138, 1103, 1436, 1972, 2898, 1146,
#           119,  102]])

#     >>> x = torch.randn(3, 4)
# >>> sorted, indices = torch.sort(x)
# >>> sorted
# tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
#         [-0.5793,  0.0061,  0.6058,  0.9497],
#         [-0.5071,  0.3343,  0.9553,  1.0960]])
# >>> indices
# tensor([[ 1,  0,  2,  3],
#         [ 3,  1,  0,  2],
#         [ 0,  3,  1,  2]])


    print("\nsorted_indexes : ", sorted_indexes)
    print("\nsorted_indexes size: ", sorted_indexes.size())

    ranks = torch.where(sorted_indexes == token_ids)[1] + 1

    print("\nranks : ", ranks)
    print("\nranks size: ", ranks.size())

    ranks = ranks.tolist()

    print("\nranks after to_list() : ", ranks)

    return ranks


def calculate_aul(model, token_ids, log_softmax, attention):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AULA or AUL).
    '''
    print("----------calculate_aul starts -----------")
    
    output = model(token_ids)

    # print("\noutput : ", output)

    logits = output.logits.squeeze(0)

    print("\nlogits : ", logits)
    print("\nlogits size : ", logits.size())    

    log_probs = log_softmax(logits)

    print("\nlog_probs :", log_probs)
    print("\nlog_probs size :", log_probs.size())

    print("\ntoken_ids size before :", token_ids.size())

    token_ids = token_ids.view(-1, 1).detach()

    print("\ntoken_ids : ", token_ids)
    print("\ntoken_ids size after :", token_ids.size())

    token_log_probs = log_probs.gather(1, token_ids)[1:-1]

    print("\ntoken_log_probs : ", token_log_probs)
    print("\ntoken_log_probs size: ", token_log_probs.size())

    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)

        print("\nattentions : ", attentions)
        print("\nattentions size: ", attentions.size())

        averaged_attentions = torch.mean(attentions, 0)

        print("\naveraged_attentions : ", averaged_attentions)
        print("\naveraged_attentions size: ", averaged_attentions.size())

        averaged_token_attentions = torch.mean(averaged_attentions, 0)

        print("\naveraged_token_attentions : ", averaged_token_attentions)
        print("\naveraged_token_attentions size: ", averaged_token_attentions.size())

        token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]

        print("\ntoken_log_probs : ", token_log_probs)
        print("\ntoken_log_probs size: ", token_log_probs.size())

    sentence_log_prob = torch.mean(token_log_probs)

    print("\nsentence_log_prob : ", sentence_log_prob)
    print("\nsentence_log_prob size: ", sentence_log_prob.size())
   
    score = sentence_log_prob.item()

    print("\nscore : ", score)

    ranks = get_rank_for_gold_token(log_probs, token_ids)

    print("\nranks : ", ranks)

    return score, ranks


def calculate_cps(model, token_ids, spans, mask_id, log_softmax):
    '''
    Given token ids of a sequence, return the summed log probability of
    masked shared tokens between sequence pair (CPS).
    '''
    print("----------calculate_cps starts -----------")

    print("\nspans size: ", len(spans))

    spans = spans[1:-1]
    print("\nspans : ", spans)
    print("\nspans size after removing 101 and 102: ", len(spans))

    masked_token_ids = token_ids.repeat(len(spans), 1)
    print("\nmasked_token_ids : ", masked_token_ids)
    print("\nmasked_token_ids size: ", masked_token_ids.size())

    masked_token_ids[range(masked_token_ids.size(0)), spans] = mask_id

    print("\nmasked_token_ids : ", masked_token_ids)
    print("\nmasked_token_ids size: ", masked_token_ids.size())

    hidden_states = model(masked_token_ids)

    print("\nhidden_states : ", hidden_states)

    hidden_states = hidden_states[0]   

    print("\nhidden_states : ", hidden_states)
    print("\nhidden_states size: ", hidden_states.size())

    token_ids = token_ids.view(-1)[spans]
       
    print("\ntoken_ids : ", token_ids)
    print("\ntoken_ids size: ", token_ids.size())

    log_probs = log_softmax(hidden_states[range(hidden_states.size(0)), spans, :])

    print("\nlog_probs : ", log_probs)
    print("\nlog_probs size: ", log_probs.size())

    span_log_probs = log_probs[range(hidden_states.size(0)), token_ids]
    
    print("\nspan_log_probs : ", span_log_probs)
    print("\nspan_log_probs size: ", span_log_probs.size())

    score = torch.sum(span_log_probs).item()
    
    print("\nscore : ", score)

    ranks = get_rank_for_gold_token(log_probs, token_ids.view(-1, 1))
    
    print("\nranks : ", ranks)

    return score, ranks


def calculate_sss(model, token_ids, spans, mask_id, log_softmax):
    '''
    Given token ids of a sequence, return the averaged log probability of
    masked diffirent tokens between sequence pair (SSS).
    '''
    masked_token_ids = token_ids.clone()

    print("\nmasked_token_ids : ", masked_token_ids)
    print("\nmasked_token_ids size: ", masked_token_ids.size())

    masked_token_ids[:, spans] = mask_id

    print("\nmasked_token_ids : ", masked_token_ids)
    print("\nmasked_token_ids size: ", masked_token_ids.size())
    
    hidden_states = model(masked_token_ids)

    print("\nhidden_states : ", hidden_states)

    hidden_states = hidden_states[0].squeeze(0)

    print("\nhidden_states aft squeeze: ", hidden_states)

    token_ids = token_ids.view(-1)[spans]

    print("\ntoken_ids : ", token_ids)
    print("\ntoken_ids size: ", token_ids.size())

    log_probs = log_softmax(hidden_states)[spans]

    print("\nlog_probs : ", log_probs)
    print("\nlog_probs size: ", log_probs.size())

    span_log_probs = log_probs[:,token_ids]

    print("\nspan_log_probs : ", span_log_probs)
    print("\nspan_log_probs size: ", span_log_probs.size())

    score = torch.mean(span_log_probs).item()

    if log_probs.size(0) != 0:
        ranks = get_rank_for_gold_token(log_probs, token_ids.view(-1, 1))
    else:
        ranks = [-1]

    print("\nscore : ", score)
    print("\nranks : ", ranks)

    return score, ranks


def main(args):
    '''
    Evaluate the bias in masked language models.
    '''
    print("--------------------main function starts-----------------")

    tokenizer, model = load_tokenizer_and_model(args)
    total_score = 0
    stereo_score = 0

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    mask_id = tokenizer.mask_token_id
    
    print("\nmask_id : ", mask_id)

    log_softmax = torch.nn.LogSoftmax(dim=1)
    
    print("\nlog_softmax : ", log_softmax)

    vocab = tokenizer.get_vocab()
    
    print("\nvocab : ", vocab)

    count = defaultdict(int)
    
    print("\ncount : ", count)

    scores = defaultdict(int)
    
    print("\nscores : ", scores)

    all_ranks = []
    data = []

    with open(f'data/{args.data}_eval.json') as f:
        inputs = json.load(f)
        total_num = len(inputs)
    
        print("\ntotal_num : ", total_num)

        for input in tqdm(inputs):
            bias_type = input['bias_type']

            print("\nbias_type : ", bias_type)

            count[bias_type] += 1

            print("\ncount[bias_type] : ", count[bias_type])

            pro_sentence = input['stereotype']

            print("\npro_sentence : ", pro_sentence)

            pro_token_ids = tokenizer.encode(pro_sentence, return_tensors='pt')
            
            print("\npro_token_ids : ", pro_token_ids)
            print("\npro_token_ids size: ", pro_token_ids.size())

            anti_sentence = input['anti-stereotype']

            print("\nanti_sentence : ", anti_sentence)

            anti_token_ids = tokenizer.encode(anti_sentence, return_tensors='pt')
            
            print("\nanti_token_ids : ", anti_token_ids)
            print("\nanti_token_ids size: ", anti_token_ids.size())

            with torch.no_grad():
                if args.method == 'aula' or args.method == 'aul':

                    print("-------------------aul score---------------")

                    attention = True if args.method == 'aula' else False
                    pro_score, pro_ranks = calculate_aul(model, pro_token_ids, log_softmax, attention)
            
                    print("\npro_score : ", pro_score,"\npro_ranks : ", pro_ranks)    
                
                    anti_score, anti_ranks = calculate_aul(model, anti_token_ids, log_softmax, attention)

                    print("\nanti_score : ", anti_score,"\nanti_ranks : ", anti_ranks)    

                elif args.method == 'cps':
                    
                    print("-------------------cps score---------------")

                    pro_spans, anti_spans = get_span(pro_token_ids[0],
                                                     anti_token_ids[0], 'equal')
                    print('\nAfter get_span')
                    print("\npro_spans : ", pro_spans, "\nanti_spans", anti_spans)

                    pro_score, pro_ranks = calculate_cps(model, pro_token_ids, pro_spans,
                                              mask_id, log_softmax)
                    print("\nAfter calculate_cps")
                    print("\npro_score : ", pro_score, "\npro_ranks : ", pro_ranks)

                    anti_score, anti_ranks = calculate_cps(model, anti_token_ids, anti_spans,
                                               mask_id, log_softmax)
                    
                    print('\nanti_score : ', anti_score, '\nanti_ranks : ', anti_ranks)

                    pro_score = round(pro_score, 3)
            
                    print("\npro_score : ", pro_score)
            
                    anti_score = round(anti_score, 3)     

                    print("\nanti_score : ", anti_score)

                    data.append([anti_sentence, pro_sentence, anti_score, pro_score])
                elif args.method == 'sss':
                    pro_spans, anti_spans = get_span(pro_token_ids[0],
                                                     anti_token_ids[0], 'diff')
                    pro_score, anti_ranks = calculate_sss(model, pro_token_ids, pro_spans,
                                              mask_id, log_softmax)
                    anti_score, pro_ranks = calculate_sss(model, anti_token_ids, anti_spans,
                                               mask_id, log_softmax)

            all_ranks += anti_ranks   
            print("\nFor anti ranks")
            print("\nall_ranks : ", all_ranks)

            all_ranks += pro_ranks   
            print("\nWith pro_ranks too")
            print("\nall_ranks : ", all_ranks)

            total_score += 1   
                     
            print("\ntotal_score : ", total_score)
        
            if pro_score > anti_score:
                stereo_score += 1

                print("\nstereo_score : ", stereo_score)

                scores[bias_type] += 1

                print("\nscores : ", scores)

    fw = open(args.output, 'w')
    bias_score = round((stereo_score / total_score) * 100, 2)   
                     
    print("\nbias_score : ", bias_score)

    fw.write(f'Bias score: {bias_score}\n')
    for bias_type, score in sorted(scores.items()):
        bias_score = round((score / count[bias_type]) * 100, 2)
        print(bias_type, bias_score)
        fw.write(f'{bias_type}: {bias_score}\n')
    all_ranks = [rank for rank in all_ranks if rank != -1]   
                     
    print("\nall_ranks : ", all_ranks)
    
    accuracy = sum([1 for rank in all_ranks if rank == 1]) / len(all_ranks)       
    accuracy *= 100

    print("\naccuracy : ", accuracy)

    print(f'Accuracy: {accuracy:.2f}')
    fw.write(f'Accuracy: {accuracy:.2f}\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
# python evaluate.py --data cp --output data\cp_output.txt --model bert --method aul