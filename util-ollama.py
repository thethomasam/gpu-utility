import os
import subprocess
import time
import ollama 
import datetime
import pandas as pd
import numpy as np
from datasets import load_dataset
import time
import nltk
import random
start = time.time()


def chat_with_bot(text):
    # response = ollama.generate(model='mistral', messages=[
    #   {
    #     'role': 'user',
    #     'content': text,
    #     'stream': False
    #   },
    # ])
    response= ollama.generate(model='mistral', prompt=text)
    return response



def record_resource_utilisation(prompts_by_bin,sample_size):
    all_records=[]
    for i in prompts_by_bin.items():
      prompts=random.sample(list(i)[1],sample_size)
      bin_number=list(i)[0]
      print('Started logging compute utilisation')
      os.system('rm -f log_compute.csv')
      logger_fname = 'log_compute.csv'
      logger_pid = subprocess.Popen(
        ['python3', 'log_gpu_cpu_stats.py',
        logger_fname,
        '--loop',  '0.2',  # Interval between measurements, in seconds (optional, default=1)
        ])
      response_inference=[]
      for i in prompts:
        start=time.time()
        response=chat_with_bot(i)
        end=time.time()
        infererence_time=end-start
        response_inference.append(infererence_time)
      logger_pid.terminate()
      # print('response done')
      # print(response_inference)
      gpu_util=pd.read_csv('./log_compute.csv')

      record={}
      record['CPU (%)']=gpu_util[['CPU (%)']].mean()
      record['RAM (%)']=gpu_util[['RAM (%)']].mean()
      record['Swap (%)']=gpu_util[['Swap (%)']].mean()
      record['0:GPU (%)']=gpu_util[['0:GPU (%)']].mean()
      record['0:Mem (%)']=gpu_util[['0:Mem (%)']].mean()
      record['0:Temp (C)']=gpu_util[['0:Temp (C)']].mean()
      record['Token Size']=bin_number
      record['Average_reponse_time']=np.mean(response_inference)
      all_records.append(record)
      print('done ' +str(bin_number))
    df=pd.DataFrame(all_records)
    print(df)
    
    

   
    
if __name__ == "__main__":
  df=pd.read_csv('./prompt-bin.csv')
  prompts_by_bin = df.groupby('bin_number')['prompt'].apply(list)
  # print(prompts_by_bin)
  record_resource_utilisation(prompts_by_bin,10)
