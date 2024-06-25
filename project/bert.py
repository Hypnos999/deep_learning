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
    fake_news = pd.read_csv('data/Fake.csv')
    fake_news['label'] = 0

    true_news = pd.read_csv('data/True.csv')
    true_news['label'] = 1

    news = pd.concat([fake_news, true_news])
    df = news[['title', 'label']]
    # df = pd.concat([fake_news[:1000], true_news[:1000]])[['title', 'label']]

    x_train, x_test, y_train, y_test = train_test_split(df['title'], df['label'], test_size=0.1, random_state=42, shuffle=True, stratify=df['label'])

    # Use the bert preprocessor and bert encoder from tensorflow_hub
    # bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    # bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', trainable=True)

    bert_preprocess = hub.KerasLayer( "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1", trainable=True)

    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='news')
    bert_processed = bert_preprocess(input_layer)
    bert_output = bert_encoder(bert_processed)
    hidden = bert_output['pooled_output']
    # hidden = tf.keras.layers.Dropout(0.2)(hidden)
    # hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(hidden)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output])

    # Compile model on adam optimizer, binary_crossentropy loss, and accuracy metrics
    epochs = 5
    # tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = tf.metrics.BinaryAccuracy()

    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    model.summary()

    # Train model on 5 epochs
    # history = model.fit(x_train, y_train, epochs=epochs, batch_size=32*2, validation_split=0.2)

    # Evaluate model on test data
    # loss, accuracy = model.evaluate(x_test, y_test)

    # print(f'Test Loss: {loss}')
    # print(f'Test Accuracy: {accuracy}')

    model.load_weights('bert.weights.h5')
    # model.save_weights('bert.weights.h5')

    # # "Accuracy"
    # plt.plot(history.history['binary_accuracy'])
    # plt.plot(history.history['val_binary_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    #
    # # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    print(y_pred)

    # # Confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    # plt.colorbar()
    # plt.xticks(np.arange(2), ['fake', 'true'], rotation=45)
    # plt.yticks(np.arange(2), ['fake', 'true'])

    # # Add labels and title
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.title('Confusion Matrix')
    #
    # # Add values to the matrix
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         plt.text(j, i, format(cm[i, j], 'd'),
    #                  ha="center", va="center",
    #                  color="white" if cm[i, j] > thresh else "black")
    #
    # plt.tight_layout()
    # plt.show()

    target_names = ['fake news', 'real news']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Results
    # accuracy 0.94
    # val accuracy 0.95
