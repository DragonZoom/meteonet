#!/bin/bash

if ! [ -f data/README.org ]; then
    curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/data.tar.gz --output data.tar.gz
    tar xvfz data.tar.gz
    rm data.tar.gz
else
    echo "Data already downloaded: good!"
fi

