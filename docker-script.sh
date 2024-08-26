# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export TZ=America/San_Francisco
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1
export PIP_ROOT_USER_ACTION=ignore
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export GCS_RESOLVE_REFRESH_SECS=60
export GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
export GCS_METADATA_REQUEST_TIMEOUT_SECS=300
export GCS_READ_REQUEST_TIMEOUT_SECS=300
export GCS_WRITE_REQUEST_TIMEOUT_SECS=600
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp

# Update and install dependencies
apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common \
  libglew-dev x11-xserver-utils xvfb \
  && apt-get clean

# Set up Python 3.11
add-apt-repository ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
python3.11 -m venv ./venv --upgrade-deps
source ./venv/bin/activate
pip install --upgrade pip setuptools

# Install agent requirements
pip install -r dreamerv3/requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install embodied requirements
pip install -r embodied/requirements.txt

# Set permissions
chown 1000:root . && chmod 775 .

# Install Tetris-Gymnasium
pip install opencv-python-headless
cd ..
git clone https://github.com/Max-We/Tetris-Gymnasium.git
cd ./Tetris-Gymnasium
pip install .
cd ../dreamerv3

# Login wandb
pip install wandb moviepy imageio
wandb login