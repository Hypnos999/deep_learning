import numpy as np
import pandas as pd
fake_news = pd.read_csv('data/original/Fake.csv')
fake_news['label'] = 0

true_news = pd.read_csv('data/original/True.csv')
true_news['label'] = 1

news = pd.concat([fake_news, true_news])
# df = (news[['text', 'label']]
df = news.dropna().drop_duplicates()
print(df['subject'].value_counts())

from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='6cab58fbcd124ccab8ff2220d13b632c')

# /v2/top-headlines
# top_headlines = newsapi.get_top_headlines(q='bitcoin',
#                                           sources='bbc-news,the-verge',
#                                           category='business',
#                                           language='en',
#                                           country='us')


# /v2/top-headlines/sources
sources = newsapi.get_sources()
print(sources)
# sources = [dictt['id'] for dictt in sources['sources']]

# /v2/everything
all_articles = newsapi.get_top_headlines(
    # q='bitcoin',
    # sources='bbc-news,the-verge',
    # domains='bbc.co.uk,techcrunch.com',
    language='en',
    # from_param='2024-12-05',
    # to='2024-12-15',
    # sort_by='relevancy',
    # page=2,
    # sources=', '.join(sources)
)
import json
print(json.dumps(all_articles, indent=2))