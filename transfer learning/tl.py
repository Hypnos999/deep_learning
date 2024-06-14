from keras.src.applications import mobilenet
from keras.src import Model
from keras.src.layers import Dense, Flatten

if __name__ == '__main__':
    mobilenet_model = mobilenet.MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        ## TOP = layers di Flatten e Dense (neuroni)
        ## include_top=False ci permette di prendere solo i layer antecedenti al Flatten (layers di convoluzione, pooling, ecc.)
        weights='imagenet'
    )
    
    for layer in mobilenet_model.layers:
        layer.trainable = False
    mobilenet_model.summary()
        
    x = mobilenet_model.output
    a = Flatten()(x)
    a = Dense(units=50, activation='relu')(a)
    
    ## espansione della rete neurale
    model = Model(inputs=mobilenet_model.input, outputs=a)
    model.summary()