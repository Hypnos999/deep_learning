import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split

# from official.nlp import optimization  # to create AdamW optimizer

# BERT model selected           : https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1
# Preprocess model auto-selected: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3


if __name__ == '__main__':
    
    # Preprocessing
    import os
    print(os.getcwd())
    df = pd.read_csv('data/WELFake_Dataset.csv')
    df = df.dropna().drop_duplicates().sample(10000)
    
    x = df['text']
    y = df['label']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True, stratify=y)

    # Carica il modello BERT base
    bert_preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', trainable=False)
    bert_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', trainable=False)
    bert_model.trainable = False
    bert_preprocess.trainable = False
    output = bert_model(bert_preprocess('Ciao come stai?'))
    print(output)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = bert_preprocess(text_input)

    outputs = bert_model(encoder_inputs)
    #x = outputs['pooled_output']
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = Dense(64, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = Dense(2, activation='softmax')(x)
    model = Model(text_input, x)

    # Costruisci il modello completo
    optimizer = Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    print(model.summary())
    history = model.fit(x_train, y_train, epochs=5)

    # loss, accuracy = model.evaluate(test_dataset)
    # print(f'Test Loss: {loss:.4f}')
    # print(f'Test Accuracy: {accuracy:.4f}')

