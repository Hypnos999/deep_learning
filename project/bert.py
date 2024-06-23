import tensorflow as tf
from keras.src.layers import Dense, Dropout, Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape
from keras.src.models import Model
from keras.src.optimizers import Adam, Adadelta
from keras_nlp.src.models import BertPreprocessor, BertBackbone, BertClassifier, BertMaskedLM
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 0: Fake
    # 1: Real
    df_welfake = pd.read_csv('data/WELFake_Dataset.csv')
    df = df_welfake

    print(df.shape)

    df = df[['text', 'label']]
    df['text'] = df['text'].str.lower()
    df = df.dropna().drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.1, random_state=42, shuffle=True, stratify=df['label'])

    model = BertClassifier.from_preset('bert_tiny_en_uncased', load_weights=True, num_classes=2)

    print(model.summary())
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=16*2*2*2*2)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    model.save_weights('bert.weights.h5')

    # results:
    # acc 99%
    # val acc 98/99%
    # loss 0.02
    # val loss 0.035

