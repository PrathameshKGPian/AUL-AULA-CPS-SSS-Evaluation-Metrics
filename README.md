# AUL-AULA-CPS-SSS-Evaluation-Metrics
This repo aims to study about the various evaluation metrics like AUL/AULA, CPS and SSS to calculate biases in sense embeddings 


# Add text comparing all the three metrics.


Evaluataion metrics' output for test_data.csv with different MLMs

    BERT-base-cased (110M)(vocab - 28996 tokens)

    RoBERTa-large (355M)(vocab - 50265 tokens)
    
    ALBERT-large-v2 (17M)(vocab - 30000 tokens)

Metrics' Accuracy 

| Model   |  CPS  |  AUL  | AULA  |  SSS  |
|---------|-------|-------|-------|-------|
| BERT    | 59.29 | 79.56 | 79.56 | 00.00 |
| RoBERTa | 59.74 | 93.79 | 93.79 | 04.96 |
| ALBERT  | 45.67 | 85.64 | 85.64 | 00.12 |

Metrics' Bias Score for "her" as stereo

| Model   |  CPS  |  AUL  | AULA  |  SSS  |
|---------|-------|-------|-------|-------|
| BERT    | 38.59 | 44.80 | 16.73 | 02.39 |
| RoBERTa | 52.57 | 63.08 | 69.53 | 21.03 |
| ALBERT  | 68.34 | 84.35 | 83.87 | 85.66 | 

Metrics' Bias Score for "her" as anti-stereo

| Model   |  CPS  |  AUL  | AULA  |  SSS  |
|---------|-------|-------|-------|-------|
| BERT    | 61.41 | 55.20 | 83.27 | 97.61 |
| RoBERTa | 47.43 | 36.92 | 30.47 | 78.97 |
| ALBERT  | 31.66 | 15.65 | 16.13 | 14.34 | 


Model size can de determined from 
    https://huggingface.co/transformers/v2.4.0/pretrained_models.html

You can achieve the above results by entering the following code in the terminal:

    python evaluate.py --data cp --output data\cp_output.txt --model bert --method aul

        where, 
              model parameter can be changed into bert, roberta or albert
            & method parameter can be changed to aul, aula, cps or sss


## Improve the annotation 
Please fill out this by choosing either her or him for each occupation. Every occupation here has to be biased toward either him/her. Whichever suits most as per societal beliefs and gender stereotypes.

https://docs.google.com/spreadsheets/d/1m42typdjvYLtkQ8rb_bSkVQIkDErT3GL-L2n4Wz6edE/edit?usp=sharing
