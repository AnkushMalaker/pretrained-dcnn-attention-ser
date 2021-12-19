# TO DO
# get training params through arg parser like batch size, epochs, training_dir etc

# Barebones implementation for now, just creating model, train and save.

from tensorflow.keras import callbacks
from utils import get_dataset
from SpeechModel import SpeechModel
from tensorflow.keras.callbacks import EarlyStopping
from os import mkdir, path
import argparse
parser = argparse.ArgumentParser(description="Script to train the model as described in the paper.")
parser.add_argument("epochs", type=int,  help="Number of Epochs")

parser.add_argument("-nc", type=bool, help="Disable caching. Enabled by default.")

# args for batchsize, data_directory, validation_split, random state, etc
args = parser.parse_args()

if args.epochs:
    EPOCHS = args.epochs
else:
    EPOCHS = 10

if args.nc:
    CACHE= False
else:
    CACHE = True


MODEL_SAVE_DIR = "saved_model"

train_ds, val_ds = get_dataset("dataset", cache=CACHE)

SP = SpeechModel()
model = SP.create_model()

ESCallback = EarlyStopping(patience=5, restore_best_weights=True, verbose=True)
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[ESCallback])

if not path.exists(MODEL_SAVE_DIR):
    mkdir(MODEL_SAVE_DIR)
model.save(MODEL_SAVE_DIR + "/" + str(EPOCHS) + "epochs_SpeechModel")