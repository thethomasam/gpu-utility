import os
import subprocess
import time
import ollama 
import datetime
import pandas as pd


def chat_with_bot():
    response = ollama.chat(model='mistral', messages=[
      {
        'role': 'user',
        'content': 'Why is sky blue',
        'stream': False
      },
    ])
    return response

def record_resource_utilisation():
    os.system('rm -f log_compute.csv')
    logger_fname = 'log_compute.csv'
    logger_pid = subprocess.Popen(
      ['python3', 'log_gpu_cpu_stats.py',
      logger_fname,
      '--loop',  '0.2',  # Interval between measurements, in seconds (optional, default=1)
      ])
    print('Started logging compute utilisation')
    response=chat_with_bot()
    logger_pid.terminate()
    gpu_util=pd.read_csv('./log_compute.csv')
    print(gpu_util)


    print(response['message']['content'])
    
if __name__ == "__main__":
    
    record_resource_utilisation()
