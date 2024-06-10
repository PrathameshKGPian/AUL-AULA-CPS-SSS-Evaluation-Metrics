# AUL-AULA-CPS-SSS-Evaluation-Metrics
This repo aims to study about the various evaluation metrics like AUL/AULA, CPS and SSS to calculate biases in sense embeddings 


# Add text comparing all the three metrics.



Metrics' Accuracy with different MLMs

| Model   | CPS  | AUL | AULA  | SSS |
|---------|------|----------|------|----------|
| BERT    | 59.29 | 79.56        | 79.56    | 0.00        |
| RoBERTa | 59.74 | 93.79        | 93.79    | 4.96        |
| ALBERT  | 45.67    | 85.64        | 85.64    | 0.12        |

Metrics' Bias Score with different MLMs

| Model   | CPS  | AUL | AULA  | SSS |
|---------|------|----------|------|----------|
| BERT    | 38.59 | 44.80        | 16.73    | 02.39        |
| RoBERTa | 52.57 | 63.08        | 69.53    | 21.03        |
| ALBERT  | 68.34    | 84.35        | 83.87    | 85.66        |

You can achieve the above results by entering the following code in the terminal:

python evaluate.py --data cp --output data\cp_output.txt --model bert --method aul

where, 
      model parameter can be changed into bert, roberta or albert
    & method parameter can be changed to aul, aula, cps or sss


## Improve the annotation 
Please fill out this by choosing either her or him for each occasion. Every occupation here has to be biased toward either him/her. Whichever suits most as per societal beliefs and gender stereotypes.

https://docs.google.com/spreadsheets/d/1m42typdjvYLtkQ8rb_bSkVQIkDErT3GL-L2n4Wz6edE/edit?usp=sharing
