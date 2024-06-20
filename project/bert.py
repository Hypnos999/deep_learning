import tensorflow as tf
from keras.src.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape
from keras.src.models import Model
from keras.src.optimizers import Adam, Adadelta
from keras_nlp.src.models import BertPreprocessor, BertBackbone, BertClassifier, BertMaskedLM
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df_welfake = pd.read_csv('data/WELFake_Dataset.csv')

    df_fakenews_true = pd.read_csv('data/Fake.csv')
    df_fakenews_true['label'] = 1
    df_fakenews_false = pd.read_csv('data/True.csv')
    df_fakenews_false['label'] = 0
    df = pd.concat([df_fakenews_true, df_fakenews_false, df_welfake], join='inner', ignore_index=True)
    df = df[['text', 'label']]
    df['text'] = df['text'].str.lower()
    df = df.dropna().drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.1, random_state=42, shuffle=True, stratify=df['label'])

    preprocessor = BertPreprocessor.from_preset("bert_base_en_uncased")
    encoder = BertBackbone.from_preset("bert_base_en_uncased", load_weights=True)
    # encoder.trainable = False

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessed_inputs = preprocessor(text_input)
    encoder_outputs = encoder(preprocessed_inputs)
    x = encoder_outputs['pooled_output']
    x = Reshape((x.shape[1], 1))(x)  # reshape necessario per fornire i dati ai layer di Conv1D

    x1 = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x1 = MaxPooling1D(pool_size=5, strides=5)(x1)

    x2 = Conv1D(filters=128, kernel_size=4, activation='relu')(x)
    x2 = MaxPooling1D(pool_size=5, strides=5)(x2)

    x3 = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x3 = MaxPooling1D(pool_size=5, strides=5)(x3)

    x = Concatenate(axis=1)([x1, x2, x3])

    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=5, strides=5)(x)

    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=5, strides=5)(x)

    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=2, activation='softmax')(x)

    model = Model(text_input, x)
    # optimizer = Adam(learning_rate=5e-5)
    optimizer = Adadelta(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model = BertClassifier.from_preset('bert_tiny_en_uncased', load_weights=True, num_classes=2)

    # model.load_weights('MyFakeBERT.weights.h5')

    print(model.summary())
    history = model.fit(x_train, y_train, epochs=3, validation_split=0.2, batch_size=8)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # model.save('MyFakeBERT.h5')
    # model.save_weights('MyFakeBERT.weights.h5')
    model.save_weights('MyFakeBERTMoreDF.weights.h5')
