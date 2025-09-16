#!/bin/bash

sudo yum update -y
sudo yum install git -y
sudo yum install docker -y
sudo service docker start

# Enable Docker to start on boot
sudo chkconfig docker on

# Clone Streamlit app repository
git clone https://github.com/shirsneh/insurance_rag_chatbot.git
cd insurance_rag_chatbot

# Build Docker image
sudo docker build -t chatApp .

# Run Docker container
sudo docker run -d -p 8501:8501 chatApp

echo "Your VIA chatbot is live! Visit: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
