import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras.src.models import Sequential
from keras.src.layers import Embedding, LSTM, Dense,Input,GlobalMaxPooling1D,Dropout, Bidirectional, TextVectorization
from keras._tf_keras.keras.models import Model
from keras import optimizers
from keras._tf_keras.keras.optimizers import Adam
from keras.src.utils import to_categorical
import tensorflow as tf

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
# nltk.data.path.append("/kaggle/working/nltk_data/")


def process_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I) # Remove extra white space from text

    text = re.sub(r'\W', ' ', str(text)) # Remove all the special characters from text

    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # Remove all single characters from text

    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove any character that isn't alphabetical

    text = text.lower()

    words = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    stop_words = set(stopwords.words("english"))
    Words = [word for word in words if word not in stop_words]

    Words = [word for word in Words if len(word) > 3]

    indices = np.unique(Words, return_index=True)[1]
    cleaned_text = np.array(Words)[np.sort(indices)].tolist()

    return cleaned_text

if __name__ == "__main__":
    Fake=pd.read_csv("project/data/Fake.csv")
    true=pd.read_csv("project/data/True.csv")

    Fake['label']=0
    true['label']=1

    Fake.drop(columns=["title","date","subject"],inplace=True)
    true.drop(columns=["title","date","subject"],inplace=True)
    
    df1 = pd.read_csv("project/data/fake_or_real_news.csv")
    df1 = df1[df1['text'].apply(lambda x: isinstance(x, str) and x != '')]
    df1['label'] = df1['label'].replace({'FAKE': 0, 'REAL': 1})
    df1.drop(columns=['title'], inplace=True)
    News=pd.concat([Fake,true, df1],ignore_index=True)
    News['label'] = to_categorical(News['label'])
    
    df2 = pd.read_csv("project/data/WELFake_Dataset.csv")
    df2 = df2[df2['text'].apply(lambda x: isinstance(x, str) and x != '')]
    df2 = df2[df2['label'].apply(lambda x: x)]
    # df2['label'] = df2['label'].replace({'FAKE': 0, 'REAL': 1})
    df2.drop(columns=['title'], inplace=True)
    News=pd.concat([Fake,true, df1],ignore_index=True)
    News['label'] = to_categorical(News['label'])
    
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(News.head())
        print('')
        # print(News.info())
        News.drop_duplicates(inplace=True)
        # print('')
        print(News.info())
        
    x=News.drop('label',axis=1)
    y=News.label

    texts=list(x['text'])
    cleaned_text = [process_text(text) for text in texts]
    # print(cleaned_text[:10])

    X_train, X_test, y_train, y_test = train_test_split(cleaned_text, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    print(y_train.value_counts())
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(X_train)
    # word_idx = tokenizer.word_index  # Corrected syntax for accessing word index
    # v = len(word_idx)
    # print("the size of vocab =", v)  # Corrected spacing
    # X_train = tokenizer.texts_to_sequences(X_train)
    # X_test = tokenizer.texts_to_sequences(X_test)

    # maxlen = 150
    # X_train = pad_sequences(X_train,maxlen=maxlen)
    # X_test = pad_sequences(X_test,maxlen=maxlen)
    
    # y.value_counts()

    # inputt=Input(shape=(maxlen,))
    # learning_rate = 0.0001
    # x=Embedding(v+1,100)(inputt)
    # x = Dropout(0.5)(x)
    # x = LSTM(150,return_sequences=True)(x)
    # x = Dropout(0.5)(x)
    # x = GlobalMaxPooling1D()(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(2, activation='softmax')(x)

    # model = Model(inputt, x)
    
    
    MAX_INPUT_TOKENS = 10000
    OUTPUT_SEQUENCE_LENGTH = 400
    vectorization_layer = TextVectorization(max_tokens=MAX_INPUT_TOKENS, output_sequence_length=OUTPUT_SEQUENCE_LENGTH)
    vectorization_layer.adapt(X_train)
    
    model = Sequential([
        Input(shape=(1,), dtype=tf.string),
        vectorization_layer,
        Embedding(MAX_INPUT_TOKENS, OUTPUT_SEQUENCE_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(128)),
        Dropout(0.2),
        Dense(256, activation="relu"),
        Dense(1)
    ])
    
    model.summary()

    ## Define optimizer with specified learning rate
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    

    y_train_one_hot = to_categorical(y_train_encoded)
    y_test_one_hot = to_categorical(y_test_encoded)

    # history = model.fit(X_train, y_train_one_hot, epochs=2, validation_data=(X_test, y_test_one_hot))
    
    # ## Plot training & validation accuracy values
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # ## Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # # Evaluate the model on the test data
    # loss, accuracy = model.evaluate(X_test, y_test_one_hot)

    # print("Test Loss:", loss)
    # print("Test Accuracy:", accuracy)

    ## Save/load wehigts and model
    model.load_weights('model.weights.h5')
    # model.save('model.h5')
    # model.save_weights('model.weights.h5')

    # y_pred_probs = model.predict(X_test)
    # y_pred_labels = np.argmax(y_pred_probs, axis=1)
    # y_true_labels = np.argmax(y_test_one_hot, axis=1)
    # conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=['Fake', 'Real'], 
    #             yticklabels=['Fake', 'Real'])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

    ## best results:
    ## accuracy: 0.975
    ## loss: 0.07...
    
    ## TEST con un altro dataset
    print('\nTesting different dataset')
    df1 = pd.read_csv("project/data/WELFake_Dataset.csv")
    df1 = df1[df1['text'].apply(lambda x: isinstance(x, str) and x != '')]
    
    texts = list(df1['text'])
    xx = [process_text(text) for text in texts]
    xx = tokenizer.texts_to_sequences(xx)
    xx = pad_sequences(xx, maxlen=maxlen)
    
    yy = df1['label']
    yy = label_encoder.transform(yy)
    yy = to_categorical(yy)
    
    prediction = model.predict(xx)
    
    accuracy = np.mean(prediction.argmax(axis=1) == yy.argmax(axis=1))
    print('Accuracy:', accuracy)

    ## Evaluate the model on the test data
    # loss, accuracy = model.evaluate(xx, yy)
    # print("Test Loss:", loss)
    # print("Test Accuracy:", accuracy)
