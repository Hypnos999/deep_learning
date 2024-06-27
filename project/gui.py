import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont
import requests
import json
from eventregistry import *
#from newsapi import NewsApiClient
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

bert_preprocess = hub.KerasLayer( "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3", name="BERT_preprocessing")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1", trainable=True, name="BERT_encoder")

input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input')
bert_processed = bert_preprocess(input_layer)
bert_output = bert_encoder(bert_processed)
hidden = bert_output['pooled_output']
hidden = tf.keras.layers.Reshape((hidden.shape[1], 1))(hidden)
hidden = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu')(hidden)
hidden = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(hidden)
hidden = tf.keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu')(hidden)
hidden = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4)(hidden)
hidden = tf.keras.layers.Flatten()(hidden)
hidden = tf.keras.layers.Dropout(0.2)(hidden)
hidden = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(hidden)
hidden = tf.keras.layers.Dropout(0.2)(hidden)
hidden = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(hidden)
output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(hidden)

model = tf.keras.Model(inputs=[input_layer], outputs=[output])

epochs = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metrics = tf.metrics.BinaryAccuracy()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#model.summary()
model.load_weights('results/bert_cnn/bert_cnn.weights.h5')

# Function to get news from the API
def get_news():
    # Replace with your API endpoint
    er = EventRegistry(apiKey="d1c4a948-80b6-43c0-9371-13b66cfa4a9f")

    # get the USA URI
    usUri = er.getLocationUri("USA")  # = http://en.wikipedia.org/wiki/United_States

    #categoryUri = er.getCategoryUri("politics"),
    q = QueryArticlesIter(sourceLocationUri=usUri)

    # obtain at most 500 newest articles or blog posts, remove maxItems to get all
    articles = []
    for art in q.execQuery(er, sortBy="date", maxItems=100):
        if art['isDuplicate']: continue
        if art['lang'] != 'eng': continue

        articles.append(art)

    return articles

# Function to predict if news is fake or not
def predict_news(news_text):
    # Preprocess the news text
    # ... (your code to preprocess the text)

    # Make prediction using your TensorFlow model
    prediction = model.predict([news_text])
    print(prediction)

    if prediction <= 0.5:
        return "Real News"
    else:
        return "Fake News"


class FakeNewsDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fake News Detector")
        self.setStyleSheet("""
               QMainWindow {
                   background-color: #f0f0f0;
               }
               QTextEdit {
                   font-family: Arial;
                   font-size: 14px;
                   background-color: #ffffff;
                   border: 1px solid #cccccc;
               }
               QPushButton {
                   font-family: Arial;
                   font-size: 14px;
                   background-color: #4CAF50;
                   color: #ffffff;
                   padding: 10px 20px;
                   border: none;
                   border-radius: 4px;
               }
           """)

        # Create a text area to display news
        self.news_text = QTextEdit()
        font = QFont("Arial", 16)
        self.news_text.setFont(font)

        # Create a button to fetch news from the API and predict if it's fake or not
        fetch_and_predict_button = QPushButton("Fetch and Predict")
        fetch_and_predict_button.clicked.connect(self.fetch_and_predict)

        # Create a layout and add the text area and button
        layout = QVBoxLayout()
        layout.addWidget(self.news_text)
        layout.addWidget(fetch_and_predict_button)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def showEvent(self, event):
        self.showMaximized()
        super().showEvent(event)

    def fetch_and_predict(self):
        self.news_text.clear()  # Clear the text area
        news_data = get_news()
        for news_item in reversed(news_data):
            print(news_item)
            self.news_text.append('Date: ' + news_item['date'] + ' ' + news_item['time'])
            self.news_text.append('From: ' + news_item['source']['title'])
            self.news_text.append('Title: ' + news_item["title"])
            prediction = predict_news(news_item["body"])
            self.news_text.append(f"AI Prediction: {prediction}\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    detector = FakeNewsDetector()
    detector.show()
    sys.exit(app.exec_())