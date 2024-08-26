# Set up Python 3.11
add-apt-repository ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
python3.11 -m venv ./venv --upgrade-deps
source ./venv/bin/activate
pip install --upgrade pip setuptools

# Install agent requirements
pip install jax[cuda]
pip install -r dreamerv3/requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

## Install embodied requirements
#pip install -r embodied/requirements.txt

# Install Tetris-Gymnasium
#apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#pip install opencv-python-headless
#cd ..
#git clone https://github.com/Max-We/Tetris-Gymnasium.git
#cd ./Tetris-Gymnasium
#pip install .
#cd ../dreamerv3
#
## Login wandb
#pip install wandb moviepy imageio
#wandb login
#
## Test JAX
#python3.11 -c "import jax; print(jax.devices())"