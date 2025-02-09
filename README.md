# language-modeling
Start runpod with A100 and 50G of volume container

apt update && apt install vim -y && apt install screen -y

install gh client with script install_gh.sh
gh auth login
clone repo


# install miniconda
using install_miniconda.sh

# install anaconda - install location `/workspace/anaconda3`
source .bashrc
conda create -n lm python=3.10
conda activate lm
pip install -r requirements.txt
