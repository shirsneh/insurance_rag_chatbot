#!/bin/bash

sudo yum update -y
sudo yum install git -y
sudo yum install docker -y
sudo service docker start

# Enable Docker to start on boot
sudo systemctl enable docker

# Clone Streamlit app repository
git clone https://github.com/shirsneh/insurance_rag_chatbot.git
cd insurance_rag_chatbot

# Build Docker image
sudo docker build -t chatApp .

# Run Docker container
sudo docker run -d -p 8501:8501 chatApp

echo "Chatbot application is now running on port 8501"
