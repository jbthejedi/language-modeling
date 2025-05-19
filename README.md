# language-modeling
Start runpod with A100 and 50G of volume container

apt update && apt install vim -y && apt install screen -y

install gh client with script install_gh.sh
gh auth login
clone repo
gh repo clone <repo_name>


# install miniconda
using install_miniconda.sh
# install env
/workspace/miniconda3/bin/conda init
source ~/.bashrc

# Create env
conda create -n lm python=3.10
conda activate lm
pip install -r requirements.txt
