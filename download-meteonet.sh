#!/bin/bash

if ! [ -f meteonet/rainmaps/y2018-M11-d9-h0-m0.npz ]; then
    # c'est la base réduite. il faut donc trouver un endroit pour stocker les 11 giga de la base complète
    curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/meteonet.tgz --output meteonet.tgz
    tar xvfz meteonet.tgz
    rm meteonet.tgz
else
    echo "Meteonet dataset already downloaded: good!"
fi

