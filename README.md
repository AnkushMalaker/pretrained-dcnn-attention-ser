# Tensorflow Implementation for "Pretrained Deep Convolution  Neural Netowkr with Attention for Speech fasdfasd"

# Training
## Example usage
`python train.py 30` to train the model for 30 epochs  

Uses librosa to read files, which needs `sndfile`. Use `sudo apt-get install libsndfile1` to install sndfile library
Bash code to move all files into the `.dataset/` directory from indivisual sub folders like `.dataset/Actor-xx`. Run these from within the `dataset` directory.
1. `find . -mindepth 2 -type f -print -exec mv {} . \;`  
2. `rm -r Actor_*`  