# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras import optimizers
"""
    If network is overfitting => decrease batch size; the vice-versa for underfitting
"""
def lstm_model(inputs, output_size, neurons, optimizer, loss, activ_func="linear",
                dropout=0.25): 
    
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model