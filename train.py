# TO DO
# get training params through arg parser like batch size, epochs, training_dir etc
# call SpeechModel.py to get model
# Train model and save model using the args from parser


# Barebones implementation for now, just creating model, train and save.

from tensorflow.keras import callbacks
from utils import get_dataset
from SpeechModel import SpeechModel
from tensorflow.keras.callbacks import EarlyStopping
from os import mkdir
import argparse
parser = argparse.ArgumentParser(description="Script to train the model as described in the paper.")
parser.add_argument("-e", "--epochs", help="Number of Epochs")

# args for batchsize, data_directory, validation_split, random state, etc
args = parser.parse_args()

if args.EPOCHS:
    EPOCHS = args.EPOCHS
else:
    EPOCHS = 10

MODEL_SAVE_DIR = "saved_model"

train_ds, val_ds = get_dataset("dataset", cache=True)

model = SpeechModel().create_model()

ESCallback = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[ESCallback])

mkdir(MODEL_SAVE_DIR)
model.save(MODEL_SAVE_DIR + "/" + EPOCHS + "epochs_SpeechModel")
