# Install DreamerV3
pip install -U -r embodied/requirements.txt
pip install -U -r dreamerv3/requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install opencv dependencies
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install Tetris-Gymnasium
cd ..
git clone https://github.com/Max-We/Tetris-Gymnasium.git
cd ./Tetris-Gymnasium
pip install .
cd ../dreamerv3

# Login wandb
wandb login