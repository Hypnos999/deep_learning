import tensorflow as tf
from keras.src.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding
from keras.src.models import Model
from keras.src.optimizers import Adam, Adadelta
from keras_nlp.src.models import BertPreprocessor, BertBackbone
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    # Preprocessing
    df = pd.read_csv('data/WELFake_Dataset.csv')
    df = df.dropna().drop_duplicates().sample(10000)
    
    x = df['text']
    y = df['label']
    
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.11, random_state=42, shuffle=True, stratify=y)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = BertPreprocessor.from_preset("bert_base_en_uncased")
    # preprocessor.trainable = False
    encoder_inputs = preprocessor(text_input)
    encoder = BertBackbone.from_preset("bert_base_en_uncased")
    # encoder.trainable = False
    encoder_outputs = encoder(encoder_inputs)
    # pooled_output = outputs["pooled_output"]  # [batch_size, 768].
    # sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

    x = encoder_outputs['pooled_output']

    x = Embedding(1000, 100,mask_zero=True)(x)

    x1 = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x1 = MaxPooling1D(pool_size=5, strides=5)(x1)

    x2 = Conv1D(filters=128, kernel_size=4, activation='relu')(x)
    x2 = MaxPooling1D(pool_size=5, strides=5)(x2)

    x3 = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x3 = MaxPooling1D(pool_size=5, strides=5)(x3)

    x = Concatenate(axis=1)([x1, x2, x3])
    x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=5, strides=5)(x)
    x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=5, strides=5)(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=2, activation='softmax')(x)

    model = Model(text_input, x)

    # Costruisci il modello completo
    # optimizer = Adam(learning_rate=5e-5)
    optimizer = Adadelta(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), batch_size=128)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    model.save('MyFakeBERT.h5')
    model.save_weights('MyFakeBERTWeights.h5')
