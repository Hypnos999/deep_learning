import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from keras.src.layers import Dense, Dropout, Input
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras_nlp.src.models import BertPreprocessor, BertBackbone
import pandas as pd
from sklearn.model_selection import train_test_split

# BERT model selected           : https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1
# Preprocess model auto-selected: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3


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
    encoder.trainable = False
    encoder_outputs = encoder(encoder_inputs)
    # pooled_output = outputs["pooled_output"]  # [batch_size, 768].
    # sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

    x = encoder_outputs['pooled_output']
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(text_input, x)

    # Costruisci il modello completo
    optimizer = Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss='crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
    model.save('MyFakeBERT')
    model.save_weights('MyFakeBERTWeights')

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

