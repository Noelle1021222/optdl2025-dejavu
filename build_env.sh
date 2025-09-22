# Name of the virtual environment
if [[ $# == 0 ]]; then
    ENV_NAME="dejavu"
else
    ENV_NAME=$1
fi

source ~/miniconda3/bin/activate
# Creating a new conda environment
echo "Clean and Creating a new conda environment named $ENV_NAME"
conda activate base
conda remove -n $ENV_NAME --all --yes
conda create -n $ENV_NAME python=3.10 -y

# Activating the environment

echo "Activating the environment: $ENV_NAME"
conda activate $ENV_NAME

# Installing libraries in requirements.txt
pip3 install --upgrade pip setuptools wheel
echo "Installing libraries in requirements.txt"
cat requirements.txt | xargs -n 1 -L 1 pip3 install

echo "Setup complete. Activate the conda environment using: conda activate $ENV_NAME"