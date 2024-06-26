import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont
import requests
import json
#from newsapi import NewsApiClient

# Load your TensorFlow model
# ... (your code to load the model)

# Function to get news from the API
def get_news():
    # Replace with your API endpoint
    api_url = "https://eventregistry.org/api/v1/article/getArticles"
    response = requests.get(api_url,  params={

        "action": "getArticles",
        "lang": [
            "eng",
        ],
        "keyword": "Barack Obama",
        "articlesPage": 1,
        "articlesCount": 100,
        "articlesSortBy": "date",
        "articlesArticleBodyLen": -1,
        "resultType": "articles",
        "articlesSortByAsc": False,
        "apiKey": "d1c4a948-80b6-43c0-9371-13b66cfa4a9f"
    })
    print(response.status_code)
    news_data = json.loads(response.text)['articles']['results']
    news_data = [e for e in news_data if not e['isDuplicate']]
    #news_data = json.loads('[{"title": "ciao"}, {"title": "ciao2"}, {"title": "ciao3"}]')
    print(news_data)
    return news_data

# Function to predict if news is fake or not
def predict_news(news_text):
    # Preprocess the news text
    # ... (your code to preprocess the text)

    # Make prediction using your TensorFlow model
    #prediction = model.predict(preprocessed_text)
    prediction = 0
    if prediction > 0.5:
        return "Real News"
    else:
        return "Fake News"


class FakeNewsDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fake News Detector")

        # Create a text area to display news
        self.news_text = QTextEdit()
        font = QFont("Arial", 14)
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
        for news_item in news_data:
            self.news_text.append(news_item["title"])
            prediction = predict_news(news_item["title"])
            self.news_text.append(f"Prediction: {prediction}\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    detector = FakeNewsDetector()
    detector.show()
    sys.exit(app.exec_())