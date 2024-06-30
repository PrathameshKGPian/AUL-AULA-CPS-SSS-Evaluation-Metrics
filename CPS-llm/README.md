To run the CPS-llm code for its data, paste the following command in VS code terminal:

    python CPS-llm\metric_autoregressive.py --input_file CPS-llm\test_data.csv --lm_model_path gpt2-xl --output_file output_autoreg

Following summary must be printed in output_autoreg.summary file

    Total examples: 837
    Num. neutral:0, % neutral: 0.0
    Test data Overall Score: 99.52
    Score Breakdown by Target of Bias:
    Category: Male
        Number of examples: 837
        Bias score against group Male: 99.52
    For pasting into spreadsheet (Order Overall, 'Male'):99.52, 99.52