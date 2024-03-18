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


def prepare_data():

    dataset = load_dataset("nvidia/HelpSteer")
    ds = load_dataset("nvidia/HelpSteer")

    train = ds['train'] # len(train) = 35331 (95%)
    val = ds['validation']     # len(val) = 1789 (5%)
    prompts=[]
    for i in train:
        prompts.append(i['prompt'])
    df=pd.DataFrame()
   
    df['prompt']=prompts
    return df

def plot_token_length_frequency(token_length_counts):
    plt.figure(figsize=(10, 6))
    token_length_counts.plot(kind='bar')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title('Token Length Frequency')

    plt.savefig('./freq.png')

def pre_process(df,q):
    df['token_len']=df.apply(lambda row: len(nltk.word_tokenize(row['prompt'])),axis=1)
    
    # Assign each row to a bin
    df['bin'] = df['bin'] = pd.qcut(df['token_len'], q=25)
    bin_max_values = df.groupby('bin')['token_len'].max()
    df['bin_number'] = df['bin'].map(bin_max_values)
    df.to_csv('./prompt-bin.csv')
    return df['bin_number'].value_counts()




if __name__ == "__main__":
    df=prepare_data()
    token_length_counts= pre_process(df,50)
   
   

    plot_token_length_frequency(token_length_counts)
  