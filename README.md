# language-modeling
Start runpod with A100 and 50G of volume container

apt update && apt install vim -y && apt install screen -y

install gh client with script install_gh.sh
gh auth login
clone repo

install anaconda - install location `/workspace/anaconda3`
source .bashrc
conda create -n lm python=3.10
conda activate lm
pip install -r requirements.txt



# install miniconda
https://docs.anaconda.com/miniconda/install/#quick-command-line-install
mkdir -p /workspace/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda3/miniconda.sh
bash /workspace/miniconda3/miniconda.sh -b -u -p /workspace/miniconda3
rm /workspace/miniconda3/miniconda.sh