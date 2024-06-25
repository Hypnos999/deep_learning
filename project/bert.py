import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

if __name__ == '__main__':
    fake_news = pd.read_csv('data/Fake.csv')
    fake_news['label'] = 0

    true_news = pd.read_csv('data/True.csv')
    true_news['label'] = 1

    news = pd.concat([fake_news, true_news])
    df = news[['title', 'label']]

    x_train, x_test, y_train, y_test = train_test_split(df['title'], df['label'], test_size=0.1, random_state=42, shuffle=True, stratify=df['label'])

    # Use the bert preprocessor and bert encoder from tensorflow_hub
    # bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    # bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', trainable=True)

    bert_preprocess = hub.KerasLayer( "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")
    # encoder = hub.load("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-8-h-128-a-2/2")
    # bert_encoder = hub.KerasLayer(encoder.mlm, trainable=True)
    bert_encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-10-h-128-a-2/2", trainable=True)

    # Input Layers
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='news')

    # BERT layers
    processed = bert_preprocess(input_layer)
    output = bert_encoder(processed)

    # Fully Connected Layers
    # layer = tf.keras.layers.Dropout(0.2, name='dropout')(output['mlm_logits'])
    # layer = tf.keras.layers.Dropout(0.2, name='dropout')(output['pooled_output'])
    # layer = tf.keras.layers.Dense(10, activation='relu', name='hidden')(layer)
    # layer = tf.keras.layers.Dropout(0.2, name='dropout')(output['pooled_output'])
    # layer = tf.keras.layers.Dense(10, activation='relu', name='hidden')(layer)
    layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(layer)

    model = tf.keras.Model(inputs=[input_layer], outputs=[layer])
    # Compile model on adam optimizer, binary_crossentropy loss, and accuracy metrics
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Train model on 5 epochs
    history = model.fit(x_train, y_train, epochs=5, batch_size=32*2, validation_split=0.2)

    # Evaluate model on test data
    loss, accuracy = model.evaluate(x_test, y_test)

    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    model.save_weights('bert.weights.h5')

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

    # Results
    # accuracy 0.83
    # val accuracy 0.85