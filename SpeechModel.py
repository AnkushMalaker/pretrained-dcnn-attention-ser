from functools import reduce
from typing import Tuple
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow import reduce_mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class SpeechModel:
    
    def __init__(self) -> None:
    # def __init__(self, input_shape: Tuple) -> None:
        # self.input_shape = input_shape
        print("Downloading ResNet Weights")
        self.resnet_layer = ResNet50V2(
            include_top=False, weights='imagenet', input_shape=(64,64,3)
        )
        self.lossFn = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(1e-4)

    def create_conv_model(self):
        conv_input_layer = L.Input((64, 64, 3))
        resnet_output = self.resnet_layer(conv_input_layer)
        gap = L.GlobalAveragePooling2D()(resnet_output)
        
        conv_model = Model(conv_input_layer, gap)

        return conv_model

    def create_model(self):
        # td_input_layer = L.Input(self.input_shape)
        td_input_layer = L.Input([14,64,64,3])
        conv_model = self.create_conv_model()
        td_conv_layer = L.TimeDistributed(conv_model)(td_input_layer)
        td_bilstm = L.Bidirectional(L.LSTM(64, return_sequences=True))(td_conv_layer)
        td_bilstm = L.Bidirectional(L.LSTM(64, return_sequences=True))(td_bilstm)

        attn = L.Attention()([td_bilstm, td_bilstm])

        attn = reduce_mean(attn, axis=-2)
        td_dense = L.Dense(128, activation = 'swish')(attn)
        td_dense = L.Dropout(0.25)(td_dense)
        td_dense = L.Dense(128, activation = 'swish')(td_dense)
        td_dense = L.Dropout(0.25)(td_dense)



        td_output_layer = L.Dense(7)(td_dense)

        td_model = Model(td_input_layer, td_output_layer)

        # Compile here
        td_model.compile(
            optimizer = self.optimizer,
            loss = self.lossFn,
            metrics = ['acc']
        )
        return td_model


if __name__=='__main__':
    sp = SpeechModel()
    model = sp.create_model()
    print(model.summary())