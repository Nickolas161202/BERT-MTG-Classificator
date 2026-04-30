import pandas as pd
import json
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
with open('card_dataset.json', encoding='utf-8') as f:
    tmp =  json.load(f)


df = pd.DataFrame.from_dict(tmp, orient='index')
df = df.reset_index().rename(columns={'index': 'card_name'}) #switch the index to a number instead of the card's name

df = df[df["tags"].apply(lambda x: isinstance(x, list))]

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["tags"])

msss = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)

for train_idx, val_idx in msss.split(df, Y):
    X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]
    
