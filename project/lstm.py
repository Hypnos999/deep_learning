import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape, Bidirectional
from keras.src.models import Model
from keras.src.optimizers import Adam, Adadelta
from keras_nlp.src.models import BertPreprocessor, BertBackbone, BertClassifier, BertMaskedLM
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('other dataset/FakeNewsNet.csv')
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df.drop(["index", "news_url", "source_domain", "tweet_num"], axis=1, inplace=True)
    print(df.head())

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
    df = pd.concat((df, new_df), axis=1)

    x_train, x_test, y_train, y_test = train_test_split(df[cols], df['real'], test_size=0.1, random_state=42,
                                                        shuffle=True, stratify=df['real'])

    model = Sequential(
        layers=(
            Embedding(VOCAB_SIZE, DIMENSION, input_length=len(cols)),
            Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(tf.keras.layers.LSTM(128)),
            Dropout(0.2),
            Dense(2, activation="softmax")
        )
    )

    model.compile(optimizer="adam", loss="crossentropy", metrics=["accuracy"])
    print(model.summary())
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=16*2*2*2*2)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    model.save_weights('lstm.weights.h5')

    # "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # results:
    # acc 97%
    # val acc 83.5%
    # loss 0.27
    # val loss 0.39

