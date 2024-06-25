import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape, Bidirectional
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
from matplotlib import pyplot as plt
tf.get_logger().setLevel('ERROR')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

if __name__ == '__main__':
    fake_news = pd.read_csv('data/Fake.csv')
    fake_news['label'] = 0

    true_news = pd.read_csv('data/True.csv')
    true_news['label'] = 1

    news = pd.concat([fake_news, true_news])
    df = news[['title', 'label']]

    stopwords = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemTitles(title):
        words = nltk.word_tokenize(title)
        words = [re.sub("[^a-zA-Z0-9]", "", i).lower().strip() for i in words]
        words = [lemmatizer.lemmatize(i) for i in words if i not in stopwords]
        title = " ".join(words)
        return title
    df["title"] = df["title"].apply(lemTitles)
    print(df.head())

    VOCAB_SIZE = 10000
    DIMENSION = 100
    MAXLEN = 20
    cols = [i for i in range(0, MAXLEN)]

    def oneHot(title):
        return tf.keras.preprocessing.text.one_hot(title, VOCAB_SIZE)
    df["title"] = df["title"].apply(oneHot)
    print(df.head())

    new_df = pd.DataFrame(tf.keras.utils.pad_sequences(df["title"], padding="pre", maxlen=MAXLEN))
    # df = pd.concat((df, new_df), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(new_df, df['label'], test_size=0.1, random_state=42,
                                                        shuffle=True, stratify=df['label'])
    print(x_train[:10])
    model = Sequential(
        layers=(
            Embedding(VOCAB_SIZE, DIMENSION, input_length=len(cols)),
            Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(tf.keras.layers.LSTM(128)),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        )
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(model.summary())
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=16*2*2*2*2)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    model.save_weights('lstm.weights.h5')

    # results
    # acc 99%
    # val acc 94%
    # loss 0.02
    # val loss 0.017
