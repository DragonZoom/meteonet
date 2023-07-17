#!/bin/bash

if ! [ -f data/.full_dataset ]; then
    echo 'Download Meteonet full dataset...'
    curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/meteonet.tgz --output meteonet.tgz

    echo 'Extract archive...'
    tar xfz meteonet.tgz
    rm meteonet.tgz

    echo 'Reorganize datatset...'
    mv data/rainmap data/rainmaps
    for y in 16 17; do
	for M in {1..12}; do
	    mv data/rainmaps/train/y20$y-M$M-*.npz data/rainmaps/
	done
    done
    for M in {1..12}; do
	mv data/rainmaps/val/*-M$M-*.npz data/rainmaps/
    done
    rm -rf data/rainmaps/{train,val}

    mv data/wind data/windmaps
    rm -rf data/windmaps/{U,V}/PPMatrix

else
    echo "Meteonet full dataset already downloaded: good!"
fi
