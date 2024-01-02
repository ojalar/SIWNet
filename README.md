# SIWNet

Implementation of SIWNet from the paper "Enhanced Winter Road Surface Condition Monitoring with Computer Vision".

## Installation

Run the script with:
```
pip3 install -r requirements
```

## Usage

To train and validate, run:
```
python3 train.py -p <path-to-params> -tr <path-to-train-csv> -v <path-to-val-csv> -s <path-to-save-directory> -n <name-of-run>
```
To train and test, run:
```
python3 train.py -p <path-to-params> -tr <path-to-train-csv> -v <path-to-val-csv> -te <path-to-test-csv> -s <path-to-save-directory> -n <name-of-run>
```
For inference, run:
```
python3 inference.py -wb <path-to-basenet-weights> -wp <path-to-pihead-weights>  -i <path-to-image>
```

For testing, the model is trained with both training and validation data.

Example of the parameter format is provided in `params/example_params.json`.

## .csv data format
The training/validation/testing data should be provided as a .csv-files, which are formatted as
```
<path-to-image>, <grip-factor-value>
```
