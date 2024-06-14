import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from keras import Sequential
from keras.src.layers import Dense, Input
from keras.src.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
import numpy as np


## features enginering 
## fino a 5 parametri o con parametri ordinabili --> label encoding o encoding semplice
## tra 5 e 20 parametri --> one hot encoding
## di più --> embedding (usato spesso per il natural language processing)

def main():
    pd.set_option('future.no_silent_downcasting', True)
    names = [
        'sex',
        'length',
        'diameter',
        'height',
        'whole_weight',
        'shucked_weight',
        'viscera_weight',
        'shell_weight',
        'rings',
    ]
    df = pd.read_csv('abalone/data/abalone.data', names=names, header=None)


    ## key --> valore da rimpiazzare
    ## value --> rimpiazzo    
    valori = {
        "F": 0,
        "I": 0.5,
        "M": 1
    }
    
    # df.sex = df.sex.replace(valori) ## metodo n.1 --> deprecato
    df.sex = df.sex.apply(lambda x: valori[x]) ## metodo n.2
    # df.sex = df.sex.map(valori) ## metodo n.3
    
    df.sex = df.sex.astype(float)
    ## sex: "M", "I", "F"
    ## sex: 0, 0.5, 1

    ## separo la y dalle features (x)    
    y = df['rings'].values.astype(float)
    x = df.drop(['rings'], axis=1)
    print(x.head())
    print(x.describe())
    x = x.values
    
    ## normalizzazione validation dataset e training dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    model = Sequential(
        layers=(
            Input(shape=(8,)),
            Dense(2, activation='relu'),
            Dense(1, activation='relu')
        )
    )
    
    model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])
    
    history = model.fit(x_train, y_train, validation_split=0.25, batch_size=32, epochs=10*10)
    history = history.history
    
    y_hat = model.predict(x_test)
    y_hat = np.round(y_hat, decimals=0).astype(int)
    rmse = root_mean_squared_error(y_test, y_hat)
    
    print(f'Y HAT: {y_hat}')
    print(f'RMSE: {rmse}')
    
    xx = np.arange(1, 10*10+1, 1)
    plt.subplot(1, 2, 1)    
    plt.plot(xx, history['loss'], c='red')    
    plt.plot(xx, history['val_loss'], c='orange')    
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)    
    plt.plot(xx, history['root_mean_squared_error'], c='red')    
    plt.plot(xx, history['val_root_mean_squared_error'], c='orange') 
    plt.xlabel('Epoche')
    plt.ylabel('RMSE')
    
    plt.show()   

if __name__ == '__main__':
    main()