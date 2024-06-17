import tensorflow as tf
#! QUALSIASI IMPORT DI KERAS ROMPE IL MODELLO DI BERT PER UNA QUESTIONI DI TYPING
# from keras.src.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, GlobalMaxPooling1D
# from keras.src.models import Model
# from keras.src.optimizers import Adam, Adadelta
# from keras_nlp.src.models import BertPreprocessor, BertBackbone, BertClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFBertModel

if __name__ == '__main__':
    # Preprocessing
    # true = pd.read_csv('data/Fake.csv')
    # true['label'] = 0
    # false = pd.read_csv('data/True.csv')
    # false['label'] = 1
    # df = pd.concat([true, false])

    df = pd.read_csv('data/WELFake_Dataset.csv')
    df = df.dropna().drop_duplicates()  # .sample(10000)
    df = df.sample(df.shape[0] // 4)

    x = df['text']
    y = df['label']

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.11, random_state=42, shuffle=True, stratify=y)

    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')

    def tokenize(dataset):
        dataset = tokenizer(
            text=list(dataset),
            add_special_tokens=True,
            # max_length=100,
            truncation=True,
            padding='max_length',
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=True
        )
        return dataset

    x_train_tokenized = tokenize(x_train)
    x_val_tokenized = tokenize(x_val)
    x_test_tokenized = tokenize(x_test)

    input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name='input_mask')

    bert = TFBertModel.from_pretrained('bert-large-uncased')
    bert.trainable = False
    embeddings = bert([input_ids, input_mask])[1]  # pooler output

    print(embeddings)
    print(embeddings.shape)

    x = tf.keras.layers.Embedding(1000, 100, input_length=1000)(embeddings)
    # x = embeddings


    x1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x1 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x1)

    x2 = tf.keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu')(x)
    x2 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x2)

    x3 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x3 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x3)

    x = tf.keras.layers.Concatenate(axis=1)([x1, x2, x3])
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(x)
    # x = GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=29, strides=29)(x)
    # x = GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=368, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(units=2, activation='softmax')(x)
    # x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.models.Model([input_ids, input_mask], x)

    # Costruisci il modello completo
    # optimizer = Adam(learning_rate=5e-5)
    optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=0.001)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = tf.keras.metrics.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    print(model.summary())
    print(model.compiled_loss)
    history = model.fit(
        x = {'input_ids': x_train_tokenized['input_ids'], 'input_mask': x_train_tokenized['attention_mask']},
        y = y_train,
        validation_data= [{'input_ids': x_val_tokenized['input_ids'], 'input_mask': x_val_tokenized['attention_mask']}, y_val],
        epochs=5,
        batch_size=128
    )

    loss, accuracy = model.evaluate(x={'input_ids':x_test_tokenized['input_ids'],'input_mask':x_test_tokenized['attention_mask']}, y = y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    model.save('MyFakeBERT.h5')
    model.save_weights('MyFakeBERTWeights.h5')


