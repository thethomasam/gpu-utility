import os
import subprocess
import time
import ollama 
import datetime



def chat_with_bot(question):
    response = ollama.chat(model='mistral', messages=[
      {
        'role': 'user',
        'content': question,
        'stream': False
      },
    ])
    return response

def record_resource_utilisation(question):
    os.system('!rm -f log_compute.csv')
    logger_fname = 'log_compute.csv'
    logger_pid = subprocess.Popen(
      ['python3', 'log_gpu_cpu_stats.py',
      logger_fname,
      '--loop',  '0.2',  # Interval between measurements, in seconds (optional, default=1)
      ])
    print('Started logging compute utilisation')
    response=chat_with_bot()
    logger_pid.terminate()

    print(response['message']['content'])
    
if __name__ == "__main__":
    
    record_resource_utilisation()
