import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape
from keras.models import Model
from keras.optimizers import Adam, Adadelta
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
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', trainable=True)

    # bert_preprocess = hub.KerasLayer( "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")
    # bert_encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-10-h-128-a-2/2")

    # Input Layers
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='news')

    # BERT layers
    processed = bert_preprocess(input_layer)
    output = bert_encoder(processed)['pooled_output']
    x = Reshape((output.shape[1], 1))(output)

    x1 = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x1 = MaxPooling1D(pool_size=5, strides=5)(x1)

    x2 = Conv1D(filters=128, kernel_size=4, activation='relu')(x)
    x2 = MaxPooling1D(pool_size=5, strides=5)(x2)

    x3 = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x3 = MaxPooling1D(pool_size=5, strides=5)(x3)

    x = Concatenate(axis=1)([x1, x2, x3])

    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=5, strides=5)(x)

    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=3, strides=3)(x)

    x = Flatten()(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
    x = tf.keras.layers.Dense(10, activation='relu', name='hidden')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs=[input_layer], outputs=[x])

    # Compile model on adam optimizer, binary_crossentropy loss, and accuracy metrics
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Train model on 5 epochs
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    # Evaluate model on test data
    loss, accuracy = model.evaluate(x_test, y_test)

    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    model = Model(input_layer, x)
    # optimizer = Adam()
    # optimizer = Adadelta(learning_rate=1e-5)
    optimizer = Adadelta()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    print(model.summary())
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, batch_size=16*2*2)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    model.save_weights('fakebert.weights.h5')

    # "Accuracy"
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
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
    # acc 85%
    # val_acc 83.5%
    # loss 0.27
    # val loss 0.39





