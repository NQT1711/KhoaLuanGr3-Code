import yaml
import numpy as np
import pandas as pd
import torch
import underthesea
import os
import re
# from torch.utils.data 

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
