# Vollo security example

This repo demonstrates training a deep-learning network intrusion detection system (IDS) that can be run on an FPGA using the vollo compiler.

We will train a binary classifier to determine if a network flow-feature is ordinary traffic or an attack. We follow Ref[^1] and train a time series model on the UNSW-NB15 dataset.

[^1]: [Deep Learning for Intrusion Detection Systems (IDSs) in Time Series Data](https://www.mdpi.com/1999-5903/16/3/73)

## Steps

Work through the following from the top level of this repo.

### 1. Get the dataset

Download the csv folder from the UNSW-NB15 dataset available [here](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Unzip/extract to `data/csv`, the result of `ls data/csv` should look something include:

```
NUSW-NB15_features.csv   
UNSW-NB15_1.csv   
UNSW-NB15_3.csv   
UNSW-NB15_2.csv   
UNSW-NB15_4.csv
...      
```

### 2. Virtual environment and Vollo  

Set up a virtual environment:

```sh
python3 -m venv vollo-venv
source vollo-venv/bin/activate
pip install --upgrade pip
```

Install the vollo sdk, follow the instructions [here](https://vollo.myrtle.ai/latest/installation.html), this should create a `vollo-sdk-<version>` folder. 

```sh
pip install -r requirements.txt
pip install vollo-sdk-*/python/*.whl
```

### 3. Prepare the data

Run the data preparation script:

```sh
python data.py
```

This will take a few minutes and leave its output in `/data/processed`.

### 4. Train the model

Run the training script:

```
python train.py
```

## Results

After training the model achieves the following results on the test set:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 98.9%  |
| Precision | 95.4%  |
| Recall    | 98.4%  |
| F1-score  | 96.9%  |

For comparison the literature baseline is an F1-score ~ 90%[^1]





