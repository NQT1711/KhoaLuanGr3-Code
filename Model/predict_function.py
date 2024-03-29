import numpy as np
import pandas as pd
from text_preprocessed import preprocessing_text
from transformers import TFAutoModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, LSTM
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer
from tensorflow.data import Dataset

df_train = pd.read_csv('../Data/Pre_train_model/train.csv')

PRETRAINED_MODEL = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
MAX_SEQUENCE_LENGTH = tokenizer.model_max_length

optimizer = Adam(learning_rate=1e-6)

def create_model(optimizer):
    inputs = {
        'input_ids'     : Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_ids'),
        'token_type_ids': Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='token_type_ids'),
        'attention_mask': Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='attention_mask'),
    }
    pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
    hidden_states = pretrained_bert(inputs).hidden_states

    pooled_output = concatenate(
        tuple([hidden_states[i] for i in range(-4, 0)]),
        name = 'last_4_hidden_states',
        axis = -1
    )[:, 0, :]
    x = Dropout(0.2)(pooled_output)

    outputs = concatenate([
        Dense(
            units = 4,
            activation = 'softmax',
            name = label.replace('#', '-').replace('&', '_'),
        )(x) for label in df_train.columns[1:]
    ], axis = -1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

reloaded_model = create_model(optimizer)
reloaded_model.load_weights('model_weights/weights_3.h5')

def predict_array(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1) # sentiment values (position that have max value)

def predict_label(text):
    input = preprocessing_text(text)
    tokenized_input = tokenizer(input, padding='max_length', truncation=True)
    features = {x: [[tokenized_input[x]]] for x in tokenizer.model_input_names}
    replacements = {0: 'None', 1: 'positive', 2: 'negative', 3: 'neutral'}
    categories = ['BATTERY', 'CAMERA', 'GENERAL', 'SER&ACC', 'PERFORMANCE', 'DESIGN',
                'FEATURES', 'PRICE', 'SCREEN', 'STORAGE']
    pred = predict_array(reloaded_model, Dataset.from_tensor_slices(features))

    pred_dict = {}
    pred_dict['Review'] = text
    for i in range(10):
        pred_dict[categories[i]] = replacements[pred[0][i]]

    return pred_dict