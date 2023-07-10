#!/bin/bash

if ! [ -f data/README.org ]; then
    # c'est la base réduite. il faut donc trouver un endroit pour stocker les 13 giga de la base complète
    curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/data.tar.gz --output data.tar.gz
    tar xvfz data.tar.gz
    rm data.tar.gz

    echo "Future repo a faire (il est sur mon mac dans ~/METEONET/data)" > data/README.org
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
    echo "Data already downloaded: good!"
fi

