#!/usr/bin/env bash

set -eo pipefail

################# SETTING VM NAME AND ZONE #################
VM_NAME=extraction
ZONE=us-central1-a
GCP_PROJECT=gsextraction
VM_SSH_ALIAS="${VM_NAME}.${ZONE}.${GCP_PROJECT}"

gcloud compute config-ssh

################# COPY SCRIPT TO VM #################
# only copy the script
# scp ~/workspace/table-classification/Render_images_GS_CC2020.py ${VM_SSH_ALIAS}:~/table-classification/Render_images_GS_CC2020.py

# install rsync on the vm
ssh $VM_SSH_ALIAS 'sudo apt-get update && sudo apt-get install rsync -y'

# copy all relevant files (including logger config, pip dependencies file, ... but excluding all pickle files)
rsync -av --delete --include='final_result_250_warc_files.pkl' --exclude='*.pkl' --exclude='*.jpg' --exclude='credentials.json' --exclude='.git' ~/workspace/table-classification ${VM_SSH_ALIAS}:~/
rsync -av --delete --exclude='old' --exclude='*.pkl' --exclude='*.jpg' --exclude='.git' ~/workspace/table-classification ${VM_SSH_ALIAS}:~/

# install python 3.6 on the vm
ssh $VM_SSH_ALIAS << EOF
wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz
tar xvf Python-3.6.9.tgz
cd Python-3.6.9
./configure --enable-optimizations --enable-shared
# make -j8
sudo make altinstall
python3.6 --version
EOF

################# SSH INTO VM, INSTALL DEPENDENCIES AND START SCRIPT #################
ssh $VM_SSH_ALIAS << EOF
sudo apt-get update
sudo apt-get install -y python3-pip wkhtmltopdf xvfb

cd ~/table-classification
echo -e "\e[32mInstalling dependencies\e[0m"
pip3 install -r requirements.txt

echo -e "\e[32mStarting python script. SSH connection can be closed...\e[0m"
nohup python3.6 Extend_GS_CC2020.py &
EOF


################# TRYING TO COPY EXTRACTION RESULTS BACK TO LOCAL MACHINE #################
scp ${VM_SSH_ALIAS}:~/table-classification/gs_new_log_part0_final.pkl ~/workspace/table-classification/final_result.pkl