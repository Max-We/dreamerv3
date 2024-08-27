# Install DreamerV3
pip install -U -r dreamerv3/requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
#export XLA_CLIENT_MEM_FRACTION=0.8
pip install -U -r embodied/requirements.txt

# Install opencv dependencies
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install Tetris-Gymnasium
cd ..
git clone https://github.com/Max-We/Tetris-Gymnasium.git
cd ./Tetris-Gymnasium
pip install .
cd ../dreamerv3

# Login wandb
pip install wandb moviepy imageio
wandb login
