import os
import subprocess
import time
import ollama 
import datetime
import pandas as pd
from datasets import load_dataset
import time
import nltk
start = time.time()


def chat_with_bot(text):
    response = ollama.chat(model='mistral', messages=[
      {
        'role': 'user',
        'content': text,
        'stream': False
      },
    ])
    return response



def record_resource_utilisation(prompts_by_bin):
    for i in prompts_by_bin.items():
      prompts=list(i)[1]
      print(len(prompts))
      bin_number=list(i)[0]
      print('Started logging compute utilisation')
      os.system('rm -f log_compute.csv')
      logger_fname = 'log_compute.csv'
      logger_pid = subprocess.Popen(
        ['python3', 'log_gpu_cpu_stats.py',
        logger_fname,
        '--loop',  '0.8',  # Interval between measurements, in seconds (optional, default=1)
        ])
      for i in prompts:
        response=chat_with_bot(i)

      logger_pid.terminate()
      print('Done Logging')
      break
    
    

   
    
if __name__ == "__main__":
  df=pd.read_csv('./prompt-bin.csv')
  prompts_by_bin = df.groupby('bin_number')['prompt'].apply(list)
  # print(prompts_by_bin)
  print(len(prompts_by_bin[16]))
  # record_resource_utilisation(prompts_by_bin)
