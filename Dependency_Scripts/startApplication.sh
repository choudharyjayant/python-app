#!/bin/bash
# dir=$(dirname "$0")
# version=$(cat ${dir}/../buildNumber.txt)

# aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 619942913628.dkr.ecr.ap-south-1.amazonaws.com
# docker pull 619942913628.dkr.ecr.ap-south-1.amazonaws.com/devbackup-apt:question-genrator-api-$version
# docker run -d -p 5002:5000 --name devetae 619942913628.dkr.ecr.ap-south-1.amazonaws.com/devbackup-apt:question-genrator-api-$version
# sleep 20
cd /home/ubuntu/accelerator/question-generator-api/
sleep 10
nohup python3 main.py >/dev/null 2>&1 &
sleep 30