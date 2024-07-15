#!/bin/bash

containers=$(docker ps -qa)

if [ "$containers" = '' ]; then
    echo "No Containers"
else
    docker rm -f $(docker ps -qa)
fi

# images=$(docker images -q)

# if [ "$images" = '' ]; then
#     echo "No Images"
# else
#     docker rmi -f $(docker images -q)
# fi
