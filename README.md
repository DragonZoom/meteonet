# FsrGAN for Meteonet dataset

## Structure of this repository 

```plaintext
download-meteonet.sh   # script that downloads meteonet dataset
meteonet/              # torch meteonet functions
    utilities.py        # various functions
    loader.py           # the data loader (MeteonetDataset and MeteonetDatasetChunked classes)
    samplers.py         # oversampling routines
    filesets.py         # split meteonet files in train/val/test
    plots.py            # plot functions
models/                # U-Net, FsrGAN (With and without fusion)
trainers/              # Subroutines for training
train-*.py             # training procedures
eval-*.py              # evaluation procedures
tests/*.py             # various unity tests
```

## How to use this repository

### Download the Meteonet dataset

To download the Meteonet dataset, you can use the `download-meteonet.sh` script. This script will download the dataset in the `data` folder. You can download full dataset (rainmaps + wind) or reduced (only the rainmaps):

```bash
./download-meteonet.sh
```

To prepare chunked dataset (required by FsrGAN training script), you can use the `script-data-chunked.py` script. This script will prepare the chunked dataset in the `data-chunked` folder. You can specify compress ot not (not by default), include wind or not(without wind by default). See `--help` for more options.

```bash
python ./script-data-chunked.py
```


### Train the U-Net model

To train the U-Net model, you can use the `train-classif.py` script. This script will train the U-Net model on the Meteonet dataset. You can specify the number of epochs, the batch size, the learning rate, etc.:

```bash
python ./train-classif.py --epochs 20 --batch-size 32
```

### Train the FsrGAN model

NOTE: all training scripts have `--debug 1` option to run training on a small subset of the data. This is useful for debugging purposes. 

#### First stage (FsrNet)

To train the FsrNet model, you can use the `train-gan-stage1.py` script:

Train with rainmaps and wind:
```bash
python ./train-gan-stage1.py --epochs 20 --batch-size 16 --model "FsrGAN" --data-dir "data-chunked"
```
Train with rainmaps only:
```bash
python ./train-gan-stage1.py --epochs 20 --batch-size 16 --model "FsrGAN_radar_only" --data-dir "data-chunked"
```

#### Stage 1 inference

Before training the second stage, you need to generate first stage predictions using the trained FsrNet model. You can use the `script-first-stage-inference.py` script, that will generate the predictions in the `cache/first_stage_predictions` folder.

NOTE: It is required that `--input-len`, `--time-horizon` and `--stride` parameters are the same as the ones used to train the FsrNet model (and the same as the ones that will be used to train the FsrGAN model).

```bash
python ./script-first-stage-inference.py --model-pt "./runs/stage1/<BEST_RUN>/model_s1_last_epoch.pt" --data-dir "data-chunked" --dest-dir "cache/first_stage_predictions" --imput-len 12 --time-horizon 6 --stride 12 --butch-size 32
```

#### Second stage (FsrGAN)

To train the FsrGAN model, you can use the `train-gan-stage2.py` script:

Train with rainmaps and wind:
```bash
python ./train-gan-stage2.py --epochs 20 --batch-size 16 --model "FsrGAN" --data-dir "data-chunked" --s1-data-dir "cache/first_stage_predictions"
```
Train with rainmaps only:
```bash
python ./train-gan-stage2.py --epochs 20 --batch-size 16 --model "FsrGAN_radar_only" --data-dir "data-chunked" --s1-data-dir "cache/first_stage_predictions"
```
