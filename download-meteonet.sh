#!/bin/bash

if ! [ -f meteonet/rainmaps/y2018-M11-d9-h0-m0.npz ]; then
    curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/meteonet.tgz --output meteonet.tgz
    tar xvfz meteonet.tgz
    rm meteonet.tgz
else
    echo "Meteonet dataset already downloaded: good!"
fi

