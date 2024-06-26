import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    x_train = np.load('data/train/x_train.npy', allow_pickle=True)
    x_test = np.load('data/test/x_test.npy', allow_pickle=True)
    y_train = np.load('data/train/y_train.npy', allow_pickle=True)
    y_test = np.load('data/test/y_test.npy', allow_pickle=True)

    # BERT base
    # bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    # bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', trainable=True)

    # BERT tiny
    # bert_preprocess = hub.KerasLayer( "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3", name="BERT_preprocessing")
    # bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1", trainable=True, name="BERT_encoder")
    #
    # input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='news')
    # bert_processed = bert_preprocess(input_layer)
    # bert_output = bert_encoder(bert_processed)
    # hidden = bert_output['pooled_output']
    # hidden = tf.keras.layers.Reshape((hidden.shape[1], 1))(hidden)
    # hidden = tf.keras.layers.LSTM(256)(hidden)
    # # hidden = tf.keras.layers.Dropout(0.2)(hidden)
    # # hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
    # hidden = tf.keras.layers.Dropout(0.2)(hidden)
    # output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(hidden)
    #
    # model = tf.keras.Model(inputs=[input_layer], outputs=[output])
    model = tf.keras.models.load_model(
        'results/bert_lstm/bert_lstm.h5',
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

    # Define the early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=2,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # Print message when training is stopped
        restore_best_weights=True  # Restore the best model weights after training
    )

    # Compile model on adam optimizer, binary_crossentropy loss, and accuracy metrics
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = tf.metrics.BinaryAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    # Train model on 5 epochs
    # history = model.fit(x_train, y_train, epochs=epochs, batch_size=32*2, validation_split=0.2, callbacks=[early_stop])
    # model.save('results/bert_lstm/bert_lstm.h5')
    # model.save_weights('results/bert_lstm/bert_lstm.weights.h5')
    model.load_weights('results/bert_lstm/bert_lstm.weights.h5')

    # x = np.concatenate((x_test, x_train))
    # y = np.concatenate((y_test, y_train))
    x = x_test
    y = y_test
    y_pred = model.predict(x)
    y_pred = np.where(y_pred >= 0.5, 1, 0)

    # loss, accuracy = model.evaluate(x_test, y_test)
    # print(f'Test Loss: {loss}')
    # print(f'Test Accuracy: {accuracy}')

    # # Accuracy
    # plt.plot(history.history['binary_accuracy'])
    # plt.plot(history.history['val_binary_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('results/bert_lstm/accuracy.png')
    # plt.show()
    #
    # # Loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('results/bert_lstm/loss.png')
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
    plt.savefig('results/bert_lstm/confusion_matrix.png')
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
    plt.savefig('results/bert_lstm/classification_report.png')
    plt.show()

