import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape, Bidirectional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
from matplotlib import pyplot as plt
import numpy as np
tf.get_logger().setLevel('ERROR')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords

if __name__ == '__main__':
    x_train = np.load('data/train/x_train.npy', allow_pickle=True)
    x_test = np.load('data/test/x_test.npy', allow_pickle=True)
    y_train = np.load('data/train/y_train.npy', allow_pickle=True)
    y_test = np.load('data/test/y_test.npy', allow_pickle=True)

    VOCAB_SIZE = 10000
    DIMENSION = 100
    MAXLEN = 512
    cols = [i for i in range(0, MAXLEN)]

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index
    dict(list(word_index.items())[0:10])

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=MAXLEN, padding="post", truncating="post")
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAXLEN, padding="post", truncating="post")

    model = Sequential(
        layers=(
            Embedding(VOCAB_SIZE, DIMENSION, input_length=MAXLEN),
            Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(tf.keras.layers.LSTM(512)),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        )
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Define the early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=2,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # Print message when training is stopped
        restore_best_weights=True  # Restore the best model weights after training
    )

    # history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=16*2*2, callbacks=[early_stop])
    # model.save('results/lstm/lstm.h5')
    # model.save_weights('results/lstm/lstm.weights.h5')
    model.load_weights('results/lstm/lstm.weights.h5')

    # x = np.concatenate((x_test, x_train))
    # y = np.concatenate((y_test, y_train))
    x = x_test
    y = y_test
    y_pred = model.predict(x)
    y_pred = np.where(y_pred >= 0.5, 1, 0)

    # loss, accuracy = model.evaluate(x_test, y_test)
    # print(f'Test Loss: {loss:.4f}')
    # print(f'Test Accuracy: {accuracy:.4f}')

    # # Accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('results/lstm/accuracy.png')
    # plt.show()
    #
    # # Loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('results/lstm/loss.png')
    # plt.show()

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Reds, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(2), ['fake', 'true'], rotation=45)
    plt.yticks(np.arange(2), ['fake', 'true'])

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('results/lstm/confusion_matrix.png')
    plt.show()

    # Classification report
    target_names = ['fake news', 'real news']
    cr = classification_report(y, y_pred, target_names=target_names, digits=7)
    print(cr)
    cr = classification_report(y, y_pred, target_names=target_names, output_dict=True)
    labels = list(cr.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    # metrics = ['precision', 'recall', 'f1-score', 'support']
    metrics = ['precision', 'recall', 'f1-score']
    data = np.array([[cr[label][metric] for metric in metrics] for label in labels])
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.matshow(data, cmap='jet')
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(labels)), labels)
    # plt.colorbar(cax)

    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.title('Classification Report with Support')
    plt.savefig('results/lstm/classification_report.png')
    plt.show()
