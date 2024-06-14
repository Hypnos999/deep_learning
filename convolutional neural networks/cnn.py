from keras.src.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, ConfusionMatrixDisplay, confusion_matrix
from keras import Sequential
from keras.src.layers import Dense, Input
from keras.src.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    ## dataset di cifre disegnate a mano
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    
    ## mostra 10 immagini di 3 disgnati a mano
    # x_train_three = x_train[y_train == 3]
    # for i in range(10):
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(x_train_three[i], cmap=plt.get_cmap('gray'))
    # plt.show()
    
    ## il dataset è composto da matrici 28x28, quindi devo trasformarlo in un array 1D
    ## 28x28 = 784, creerò quindi un dataset con 60000 righe da 784 colonne
    x_train = x_train.reshape(60000, 784).astype(float) ## o x_train.reshape(60000, -1)
    x_test = x_test.reshape(10000, 784).astype(float) ## o x_test.reshape(60000, -1)
    x_test /= 255
    x_train /= 255
    
    model = Sequential(
        layers=(
            # Input(shape=(784,)),
            Dense(20, activation='relu', input_dim=784),
            Dense(5, activation='relu'),
            Dense(10, activation='softmax')
        )
    )
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # print(model.summary())
    epochs = 10
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=128, epochs=epochs)
    history = history.history
    
    ## mostra la storia dell'adestramento
    xx = np.arange(1, epochs+1, 1)
    plt.subplot(1, 2, 1)    
    plt.plot(xx, history['loss'], c='red')    
    plt.plot(xx, history['val_loss'], c='blue')    
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)    
    plt.plot(xx, history['accuracy'], c='red')    
    plt.plot(xx, history['val_accuracy'], c='blue') 
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.show()   
    
    ## predizione e sostizione dei valori da vettori ai singoli numeri predetti
    y_hat = model.predict(x_test)
    y_hat = y_hat.argmax(axis=1)
    print('Predizioni:', y_hat[:10])
    
    mean = np.mean(y_hat == y_test)
    print(f'Accuracy: {mean*100}')
    
    cm = confusion_matrix(y_test, y_hat, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot()
    plt.show()
    
    ## visualiziamo 10 campioni di numeri 5 che son stati predetti come un 3
    campioni_errati = x_test[(y_test == 5) & (y_hat == 3)][:10]
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(campioni_errati[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        # plt.title(f'Pred: {y_hat[i]}')
    plt.show()