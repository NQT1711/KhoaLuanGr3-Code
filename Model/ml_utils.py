import yaml
import numpy as np
import pandas as pd
import torch
import underthesea
import os
import re
import os
import math
import yaml
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn



def label_encoder_df(df):
    return df.replace({
        'Negative':1,
        'Neural':2,
        'Postive':3
    })

def label_encoder(label, aspects):
    y = [np.nan] * len(aspects)
    ap_stm = re.findall('{(.+?)#(\w+)}', label)
    for aspect, sentiment in ap_stm:
        idx = aspects.index(aspect)
        y[idx] = sentiment
    return y


# EVALUATION
label_map= {'not_exist': 0,
            'negative':1,
            'neutral':2,
            'positive':3}

replacements={0: 'None',
              1: 'negative',
              2: 'neutral',
              3: 'positive'}
target_names = list(map(str, replacements.values()))

def aspect_detection_eval(y_test, y_pred):

  categories= y_pred.columns
  y_test= y_test.fillna('not_exist').replace(label_map).values.tolist()
  y_pred= y_pred.fillna('not_exist').replace(label_map).values.tolist()

  aspect_test = []
  aspect_pred = []

  for row_test, row_pred in zip(y_test, y_pred):
      for index, (col_test, col_pred) in enumerate(zip(row_test, row_pred)):
          aspect_test.append(bool(col_test) * categories[index])
          aspect_pred.append(bool(col_pred) * categories[index])

  aspect_report = classification_report(aspect_test, aspect_pred, digits=4, zero_division=1, output_dict=True)
  print("## Aspect Detection Evaluate ##")
  print(classification_report(aspect_test, aspect_pred, digits=4, zero_division=1))
  



def sentiment_classification_eval(y_test, y_pred):

  categories= y_pred.columns
  y_test= y_test.fillna('not_exist').replace(label_map).values.tolist()
  y_pred= y_pred.fillna('not_exist').replace(label_map).values.tolist()

  y_test_flat = np.array(y_test).flatten()
  y_pred_flat = np.array(y_pred).flatten()
  target_names = list(map(str, replacements.values()))

  polarity_report = classification_report(y_test_flat, y_pred_flat, digits=4, output_dict=True)
  print("## Sentiment Classification Evaluate ##")
  print(classification_report(y_test_flat, y_pred_flat, target_names=target_names, digits=4))
  



def combination_eval(y_test, y_pred):

  categories= y_pred.columns
  y_test= y_test.fillna('not_exist').replace(label_map).values.tolist()
  y_pred= y_pred.fillna('not_exist').replace(label_map).values.tolist()

  aspect_polarity_test = []
  aspect_polarity_pred = []

  for row_test, row_pred in zip(y_test, y_pred):
      for index, (col_test, col_pred) in enumerate(zip(row_test, row_pred)):
          aspect_polarity_test.append(f'{categories[index]},{replacements[col_test]}')
          aspect_polarity_pred.append(f'{categories[index]},{replacements[col_pred]}')

  aspect_polarity_report = classification_report(aspect_polarity_test, aspect_polarity_pred, digits=4, zero_division=1, output_dict=True)
  print("## Combination Evaluate (Aspect Detection + Sentiment Classification) ##")
  print(classification_report(aspect_polarity_test, aspect_polarity_pred, digits=4, zero_division=1))
 