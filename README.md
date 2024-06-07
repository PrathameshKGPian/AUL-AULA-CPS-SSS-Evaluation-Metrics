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

