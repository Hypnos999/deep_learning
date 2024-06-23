import tensorflow as tf
from keras.src.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape
from keras.src.models import Model
from keras.src.optimizers import Adam, Adadelta
from keras_nlp.src.models import BertPreprocessor, BertBackbone, BertClassifier, BertMaskedLM
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('data/WELFake_Dataset.csv')
    print(df.shape)

    df.dropna(subset=['text', 'title'], inplace=True)
    df['text'] = df['title'] + ' ' + df['text']

    df = df[['text', 'label']]
    df['text'] = df['text'].str.lower()
    df = df.dropna().drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.1, random_state=42, shuffle=True, stratify=df['label'])

    preprocessor = BertPreprocessor.from_preset("bert_tiny_en_uncased")
    encoder = BertBackbone.from_preset("bert_tiny_en_uncased", load_weights=True)
    # encoder.trainable = False

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessed_inputs = preprocessor(text_input)
    encoder_outputs = encoder(preprocessed_inputs)
    x = encoder_outputs['pooled_output']
    x = Reshape((x.shape[1], 1))(x)  # reshape necessario per fornire i dati ai layer di Conv1D
    # x = Embedding(30522, 100, input_length=128, trainable=True)(x)  # Embedding layer per convertire i dati in vettori di 100 dimensioni
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
    # x = Dropout(0.2)(x)
    # x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=2, activation='softmax')(x)

    model = Model(text_input, x)
    # optimizer = Adam()
    optimizer = Adadelta(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    print(model.summary())
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=16*2*2)

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



