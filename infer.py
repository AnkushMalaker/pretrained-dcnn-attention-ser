from tensorflow.keras.models import load_model
import argparse

# Infer script to be done
parser = argparse.ArgumentParser(help="Run inference on the model. Use -h to see options.")
parser.add_argument("model_path", help="/path/to/model")
args = parser.parse_args()

MODEL_PATH = None
if args.model_path:
    MODEL_PATH = args.MODEL_PATH
else:
    MODEL_PATH = "saved_model/"
model = load_model()