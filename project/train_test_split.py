import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    fake_news = pd.read_csv('data/original/Fake.csv')
    fake_news['label'] = 0

    true_news = pd.read_csv('data/original/True.csv')
    true_news['label'] = 1

    news = pd.concat([fake_news, true_news])
    df = news[['text', 'label']].dropna().drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(
        df['text'],
        df['label'],
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=df['label']
    )
    #
    np.save('data/train/x_train.npy', x_train, allow_pickle=True)
    np.save('data/test/x_test.npy', x_test, allow_pickle=True)
    np.save('data/train/y_train.npy', y_train, allow_pickle=True)
    np.save('data/test/y_test.npy', y_test, allow_pickle=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.head())
        print(df.tail())
        print(df.shape)
        print(df.isnull().sum())
        print(df.duplicated().sum())
        print(df['label'].value_counts())
        print(df.describe())