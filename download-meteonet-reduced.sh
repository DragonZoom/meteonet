#!/bin/bash

if ! [ -f data/.reduced_dataset ]; then
    echo 'Download Meteonet reduced dataset...'
    curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/data.tar.gz --output data.tar.gz

    echo 'Extract archive...'
    tar xf data.tar.gz
    rm data.tar.gz

    echo 'Reorganize datatset...'
    mv data/Rain data/rainmaps
    for y in 16 17; do
	mv data/rainmaps/train/y20$y-*.npz data/rainmaps
    done
    mv data/rainmaps/val/*.npz data/rainmaps	
    rm -rf data/rainmaps/{train,val}

    mkdir data/windmaps
    mv data/U data/windmaps
    for y in 16 17; do
	mv data/windmaps/U/train/y20$y-*.npz data/windmaps/U
    done
    mv data/windmaps/U/val/*.npz data/windmaps/U
    rm -rf data/windmaps/U/{train,val}
    mv data/V data/windmaps
    for y in 16 17; do
	mv data/windmaps/V/train/y20$y-*.npz data/windmaps/V
    done
    mv data/windmaps/V/val/*.npz data/windmaps/V    
    rm -rf data/windmaps/V/{train,val}
else
    echo "Meteonet reduced dataset already downloaded: good!"
fi
