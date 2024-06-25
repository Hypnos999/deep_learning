# Linear algebra
import numpy as np

# Data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import subprocess
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

# Download and unzip wordnet
try:
    nltk.data.find('wordnet.zip')
except:
    nltk.download('wordnet', download_dir='/kaggle/working/')
    command = "unzip /kaggle/working/corpora/wordnet.zip -d /kaggle/working/corpora"
    subprocess.run(command.split())
    nltk.data.path.append('/kaggle/working/')

# Now you can import the NLTK resources as usual
from nltk.corpus import wordnet

import tensorflow as tf
print("The TensorFlow version is: ", tf.__version__)
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM,Bidirectional
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Observe input folders in Kaggle notebook
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


if __name__ == '__main__':
    df = pd.read_csv('other dataset/WELFake_Dataset.csv')
    print(df.shape)

    df.dropna(subset=['text', 'title'], inplace=True)
    df['text'] = df['title'] + ' ' + df['text']

    df = df[['text', 'label']]
    df['text'] = df['text'].str.lower()
    df = df.dropna().drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.1, random_state=42, shuffle=True, stratify=df['label'])
