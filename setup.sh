# Install DreamerV3
pip install .

# Install jax
pip install -U "jax[cuda12]"

# Install Tetris-Gymnasium
cd ..
git clone https://github.com/Max-We/Tetris-Gymnasium.git
cd ./Tetris-Gymnasium
pip install .
cd ../dreamerv3

# Login wandb
wandb login