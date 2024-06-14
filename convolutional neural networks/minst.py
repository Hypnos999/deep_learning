from keras import Sequential
from keras.src.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1).astype(float)
    x_test = x_test.reshape(10000, 28, 28, 1).astype(float)
    y_train = y_train.astype(int)
    x_train /= 255
    x_test /= 255
    print(y_train.dtype)
    print(x_train.dtype)
    print(x_train.shape)
    print(x_test.shape)
    
    model = Sequential(
        layers=(
            Conv2D(filters=16, kernel_size=5, activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(10, activation='softmax')
        )
    )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    epoche = 10
    history = model.fit(x_train, y_train, epochs=epoche, batch_size=128,
                        validation_split=0.2)
    y_hat = model.predict(x_test)
    y_hat = np.argmax(y_hat, axis=1)
    # np.savetxt('predizioni', y_hat)
    print(y_hat[:10])
    accuracy = np.mean(y_hat == y_test)
    print(f'Accuratezza: {accuracy*100:.2f}%')
    cm = confusion_matrix(y_test, y_hat, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(range(10)))
    disp.plot()
    plt.show()
    model.save('modello.keras')

    history = history.history
    xx = np.arange(1, epoche + 1, 1)
    plt.subplot(1, 2, 1)
    plt.plot(xx, history['loss'], c='r')
    plt.plot(xx, history['val_loss'], c='blue')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(xx, history['accuracy'], c='r')
    plt.plot(xx, history['val_accuracy'], c='blue')
    plt.xlabel('Epoche')
    plt.ylabel('Accuratezza')

    plt.show()
    
    # selezione di 5 visti come 3
    campioni_errati = x_test[(y_test == 5) & (y_hat == 3)][:10]
    print(campioni_errati)
    print(campioni_errati.shape)

    plt.figure()
    for i in range(1, 10):
        try:
            plt.subplot(3, 3, i)
            plt.imshow(campioni_errati[i].reshape(28, 28),
                    cmap=plt.get_cmap('gray'))
        except: pass ##i numeri errati sono terminati (< 10)
    plt.show()