## from https://towardsdatascience.com/fake-news-classification-with-bert-afbeee601f41

import pandas as pd 
import numpy as np 
# import torch.nn as nn
# from pytorch_pretrained_bert import BertTokenizer, BertModel
# import torch
import re
import tensorflow as tf
# from torchnlp.datasets import imdb_dataset
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# from keras_nlp.models import BertModel, BertTokenizer
from transformers import TFAutoModel, AutoTokenizer
# from transformers import BertModel

class BERTForClassification(tf.keras.Model):
    
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(2, activation='softmax')
        
    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

if __name__ == '__main__':
    ## preprocessing
    df = pd.read_csv('project/data/WELFake_Dataset.csv')
    df = df.dropna().drop_duplicates().sample(10000)
    
    # def remove_url(text):
    #     url = re.compile(r'https?://\S+|www\.\S+')
    #     return url.sub(r'',str(text))
    # df['text'] = df['text'].apply(remove_url)

    # def remove_html(text):
    #     html = re.compile(r'<.*?>')
    #     return html.sub(r'',str(text))
    # df['text'] = df['text'].apply(remove_html)

    # def remove_emoji(text):
    #     emoji_pattern = re.compile("["
    #         u"\U0001F600-\U0001F64F"  # emoticons
    #         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #         u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #     "]+", flags=re.UNICODE)
    #     return emoji_pattern.sub(r'', str(text)) # no emoji
    # df['text'] = df['text'].apply(remove_emoji)

    # df['text'] = 'TITLE: ' + df.title + '; TEXT: ' + df.text
    
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df.head(), '\n')
    #     print(df.info(), '\n')
    #     print(df.describe(), '\n')
    
    ## BERT
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = TFAutoModel.from_pretrained('bert-base-uncased')
    
        
    df['text'].apply(lambda x: tokenizer.tokenize(x, padding=True, truncation=True))
    
    x = df['text']
    y = df['label']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    # print(tokenized_text)
    
    classifier = BERTForClassification(model)

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print(classifier.summary())
    
    history = classifier.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=3,
        validation_split=0.2
    )