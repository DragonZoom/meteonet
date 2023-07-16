#!/bin/bash

if ! [ -f data/rainmaps/y2018-M2-d5-h18-m55.npz ]; then
    curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/data.tar.gz --output data.tar.gz
    tar xvfz data.tar.gz
    rm data.tar.gz
    
    mv data/Rain data/rainmaps
    mv data/rainmaps/train/*.npz data/rainmaps
    mv data/rainmaps/val/*.npz data/rainmaps
    rm -rf data/rainmaps/{train,val}
    mkdir data/windmaps
    mv data/U data/windmaps
    mv data/windmaps/U/train/*.npz data/windmaps/U
    mv data/windmaps/U/val/*.npz data/windmaps/U
    rm -rf data/windmaps/U/{train,val}
    mv data/V data/windmaps
    mv data/windmaps/V/train/*.npz data/windmaps/V
    mv data/windmaps/V/val/*.npz data/windmaps/V
    rm -rf data/windmaps/V/{train,val}
else
    echo "Meteonet dataset (reduced) already downloaded: good!"
fi

