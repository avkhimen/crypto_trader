docker build -t torch_container .
docker run -it --rm -v ${PWD}/app:/app torch_container sh
