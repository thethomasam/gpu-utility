
from locust import User, task, between,events,constant_pacing
import ollama
import json
import logging
from datetime import datetime
import pandas as pd
import nltk
class LLMUser(User):
    
    def generation(self):
        # Invoke the model
        with self.environment.events.request.measure("[Send]", "Prompt"):
            df=pd.read_csv('./train_i450_o20_30_t.csv')
            sample_prompt=df.sample(1)
            response= ollama.generate(model='mistral-benchmark', prompt=sample_prompt['prompt'].values[0])
            logging.info(sample_prompt['chat_response_length'].values[0])
            logging.info(len(nltk.word_tokenize(response['response'])))
        