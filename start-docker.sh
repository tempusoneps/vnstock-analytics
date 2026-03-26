#!/bin/bash
#
DOCKER_IMAGE="vnstock_analytics:v1.0"
#
docker container prune -f
#
if [ -z "$(docker images -q $DOCKER_IMAGE 2> /dev/null)" ]; then
    echo "Image $DOCKER_IMAGE does not exist. Building without cache..."
    docker compose build --no-cache
    docker compose up -d
else
    echo "Image $DOCKER_IMAGE exists. Starting containers..."
    docker compose up -d
fi