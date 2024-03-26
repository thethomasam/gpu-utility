import os
import subprocess
import time
import ollama 
import datetime
import pandas as pd
from datasets import load_dataset
import time
import nltk
import matplotlib.pyplot as plt

# nltk.download('punkt')
def chat_with_bot(text):
   
    response= ollama.generate(model='mistral-benchmark', prompt=text)
    return response['response']

def prepare_data():

    dataset = load_dataset("nvidia/HelpSteer")
    ds = load_dataset("nvidia/HelpSteer")

    train = pd.DataFrame(ds['train']) # len(train) = 35331 (95%)
    val = pd.DataFrame(ds['validation'])
    train['prompt_token_len']=train.apply(lambda row: len(nltk.word_tokenize(row['prompt'])),axis=1)
    train['response_token_len']=train.apply(lambda row: len(nltk.word_tokenize(row['response'])),axis=1)
    train_i10_o90_100 = train[(train['prompt_token_len'] <= 10) & (train['response_token_len'] <= 100) & (train['response_token_len'] >= 90) ]
    train_i50_o170_200 = train[(train['prompt_token_len'] <= 50) & (train['prompt_token_len'] >= 40) & (train['response_token_len'] <= 200) & (train['response_token_len'] >= 170) ]
    train_i100_o300_350 = train[(train['prompt_token_len'] <= 100) & (train['prompt_token_len'] >= 90) & (train['response_token_len'] <= 350) & (train['response_token_len'] >= 300) ]
    train_i10_o400_500 = train[(train['prompt_token_len'] <= 30)&(train['prompt_token_len'] >= 20) & (train['response_token_len'] <= 200) & (train['response_token_len'] >= 170) ]
    train_i100_o10_30 = train[(train['prompt_token_len'] >= 60) & (train['prompt_token_len'] <= 100)  & (train['response_token_len'] >= 20) & (train['response_token_len'] <= 30) ]
    train_i450_o20_30 = train[(train['prompt_token_len'] >= 450) &(train['prompt_token_len'] <= 500) & (train['response_token_len'] >= 20) & (train['response_token_len'] <= 30) ]
    train_i900_o20_30 = train[(train['prompt_token_len'] >= 900) &(train['prompt_token_len'] <= 1000) & (train['response_token_len'] >= 20) & (train['response_token_len'] <= 30) ]
    output_paths = {
    'train_i10_o90_100.csv': train_i10_o90_100,
    'train_i50_o170_200.csv': train_i50_o170_200,
    'train_i100_o300_350.csv': train_i100_o300_350,
    'train_i10_o400_500.csv': train_i10_o400_500,
    'train_i100_o10_30.csv': train_i100_o10_30,
    'train_i450_o20_30.csv': train_i450_o20_30,
    'train_i900_o20_30.csv': train_i900_o20_30,
}

# Save each DataFrame to its respective CSV file
    for filename, df in output_paths.items():
        df.to_csv(filename, index=False)

def get_response_len(df):
    df['chat_response_length']=df.apply(lambda row: len(nltk.word_tokenize(chat_with_bot(row['prompt']))),axis=1)

if __name__ == "__main__":
    prepare_data()
    df=pd.read_csv('./train_i900_o20_30.csv')

    get_response_len(df)

    
   