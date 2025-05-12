# Prototype ML Model with Gradio App & Docker
In this video we look into how to prototype a Machine Learning (ML) model using Gradio.

YouTube Link: 

## Requirements
1. Python
2. Gradio 
3. Docker


## 1. Set up Environment & Run Gradio App

### Install Required Packages
```sh
sudo apt update && sudo apt upgrade -y
```
```sh
sudo apt install python3-pip -y
```


### Create Python Virtual Environment & Activate it
```sh
python -m venv venv
source venv/bin/activate
```

```sh
pip3 install -r requirements.txt
```


### Run Gradio ML App
```sh
python app.py
```


### On your browser
```sh
http://localhost:7860
```
or 
```sh
http://PublicIP:7860
```



## 2. Build and Run the Docker Image Using the Dockerfile

### Build the Docker image
```sh
docker build -t gradio-ml-app:latest .
```

### Run the Docker Container
```sh
docker run -p 7860:7860 gradio-ml-app:latest
```

### Access Gradio ML App on your browser
```sh
http://localhost:7860
```


## Clean up

### Stop the container
```sh
Ctrl + c
```

### To delete the Docker Image
```sh
docker rmi <containerID> --force
```
