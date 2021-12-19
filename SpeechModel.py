# from typing import Tuple
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
        # input_shape should be more than 32 in h and w : (64, 64, 3)
        self.resnet_layer = ResNet50V2(
            include_top=False, weights="imagenet", input_shape=(64, 64, 3)
        )
        self.lossFn = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(1e-4)
        self.conv_model = None  # Create model on call
        self.td_model = None  # Create model on call

    def create_conv_model(self):
        conv_input_layer = L.Input((64, 64, 3))
        resnet_output = self.resnet_layer(conv_input_layer)  # (2,2,2048)
        average_pool = L.AveragePooling2D((2, 2))(resnet_output)  # (1,1,2048)
        flatten = L.Flatten()(average_pool)  # (2048)

        conv_model = Model(conv_input_layer, flatten)

        return conv_model

    def model_summary(self):
        if self.td_model is None:
            print("Create model first by calling SpeechModel.create_model()")
            return None
        return self.td_model.summary()

    def create_model(self):
        # td_input_layer = L.Input(self.input_shape)
        td_input_layer = L.Input([8, 64, 64, 3])
        self.conv_model = self.create_conv_model()
        td_conv_layer = L.TimeDistributed(self.conv_model)(
            td_input_layer
        )  # output: (8, 2048)
        td_bilstm = L.Bidirectional(L.LSTM(128, return_sequences=True))(
            td_conv_layer
        )  # (8, 256)

        # Attention layer, returns matmul(distribution, value)
        # distribution is of shape [batch_size, Tq, Tv] while value is of shape [batch_size, Tv, dim]
        # The inner dimentinons except batch_size are same, we get output of dimention [batch_size, tq, dim]
        # Here, our Query and Value dimentions are 8, 256. That is, Tv, Tq = 8 and dim = 256
        # Final output of attention layer is [batch_size, 8, 256]
        bilstm_attention_seq = L.Attention(use_scale=True)(
            [td_bilstm, td_bilstm]
        )  # (8, 256)
        bilstm_attention = reduce_mean(
            bilstm_attention_seq, axis=-2
        )  # Calculate mean along each sequence
        # There is some error in this attention layer (could be the reason loss is going to nan)

        # These dimentions are changed due to the different conv model being used.
        td_dense = L.Dense(256, activation="relu")(bilstm_attention)
        td_dense = L.Dropout(0.25)(td_dense)
        td_dense = L.Dense(128, activation="relu")(td_dense)
        td_dense = L.Dense(128, activation="relu")(td_dense)
        td_dense = L.Dropout(0.25)(td_dense)

        td_output_layer = L.Dense(8)(td_dense)

        td_model = Model(td_input_layer, td_output_layer)

        # Compile here
        td_model.compile(optimizer=self.optimizer, loss=self.lossFn, metrics=["acc"])
        self.td_model = td_model
        return self.td_model


if __name__ == "__main__":
    SP = SpeechModel()
    model = SP.create_model()
    print(model.summary())
