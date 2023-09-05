#!/bin/bash

if [ -f data/.reduced_dataset ]; then
    echo "A Meteonet dataset (reduced) as already been downloaded. Abort."
    exit
fi

if [ -f data/.full_dataset ]; then
    echo "A Meteonet dataset (full) as already been downloaded. Abort."
    exit
fi

read -p "Download reduced (1.8G) or full dataset (11G) [r/f]?" ans


case $ans in
    r|R)
	echo 'Download Meteonet reduced dataset...'
	curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/data.tar.gz --output data.tar.gz

	echo 'Extract archive...'
	tar xf data.tar.gz
	rm data.tar.gz
	    
	echo 'Reorganize dataset...'
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
	;;
    f|F)
	echo 'Download Meteonet full dataset...'
	curl https://pequan.lip6.fr/~bereziat/rain-nowcasting/meteonet.tgz --output meteonet.tgz
	    
	echo 'Extract archive...'
	tar xfz meteonet.tgz
	rm meteonet.tgz
	    
	echo 'Reorganize dataset...'
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
	;;
    *)
	echo 'Please type r or f' ;;
esac
