# Testing utils

# Load_wav
def test_loadwav():
    from utils import load_wav
    wav, sr = load_wav('03-01-05-01-02-02-10.wav')
    print(wav.dtype)
    print(sr.dtype)

# Dataset
def test_dataset():
    from utils import get_dataset
    train_ds, val_ds = get_dataset("./dataset", cache=False)

    batch, label = next(iter(train_ds))
    print(batch.shape)

# SpeechModel
def test_speechModel():
    from SpeechModel import SpeechModel
    sp = SpeechModel()
    model = sp.create_model()
    print(sp.model_summary())

if __name__=='__main__':
    print("No tests")