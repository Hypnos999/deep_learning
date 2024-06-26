import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape
from keras.models import Model
from keras.optimizers import Adam, Adadelta
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == '__main__':
    x_train = np.load('data/train/x_train.npy', allow_pickle=True)
    x_test = np.load('data/test/x_test.npy', allow_pickle=True)
    y_train = np.load('data/train/y_train.npy', allow_pickle=True)
    y_test = np.load('data/test/y_test.npy', allow_pickle=True)

    # BERT base
    # bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    # bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', trainable=True)

    # BERT tiny
    bert_preprocess = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3", name="BERT_preprocessing")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1", trainable=True, name="BERT_encoder")

    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input')
    bert_processed = bert_preprocess(input_layer)
    bert_output = bert_encoder(bert_processed)['pooled_output']
    x = Reshape((bert_output.shape[1], 1))(bert_output)

    x1 = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x1 = MaxPooling1D(pool_size=5, strides=5)(x1)

    x2 = Conv1D(filters=128, kernel_size=4, activation='relu')(x)
    x2 = MaxPooling1D(pool_size=5, strides=5)(x2)

    x3 = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x3 = MaxPooling1D(pool_size=5, strides=5)(x3)

    x = Concatenate(axis=1)([x1, x2, x3])
    #
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=5, strides=5)(x)

    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=5, strides=5)(x)

    x = Flatten()(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout2')(x)
    x = tf.keras.layers.Dense(256, activation='relu', name='hidden')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout1')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='hidden1')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs=[input_layer], outputs=[x])

    # Compile model on adam optimizer, binary_crossentropy loss, and accuracy metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metric = tf.keras.metrics.BinaryAccuracy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    model.summary()

    # Train model on 5 epochs
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    model.save_weights('fakebert.weights.h5')

    # Evaluate model on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    # Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Confusion matrix
    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    print(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(2), ['fake', 'true'], rotation=45)
    plt.yticks(np.arange(2), ['fake', 'true'])

    # Add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Add values to the matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

    target_names = ['fake news', 'real news']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # results:
    # acc 93%
    # val_acc 93%