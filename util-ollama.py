from datasets import load_dataset
import pandas as pd
ds = load_dataset("nvidia/HelpSteer")

train = pd.DataFrame(ds['train']) # len(train) = 35331 (95%)
val = ds['validation']

print(train.head())



