# Vollo security example

This repo demonstrates training a deep-learning intrusion detection system (IDS) that can be run on an FPGA using the Vollo compiler.

We will train a binary classifier to determine if a network flow-feature is ordinary traffic or an attack. We follow Ref[^1] and train a time series model on the UNSW-NB15 dataset. In deployment it makes sense to train on lower-level packet features however, this requires more train-time compute not suitable for an example. 

[^1]: [Deep Learning for Intrusion Detection Systems (IDSs) in Time Series Data](https://www.mdpi.com/1999-5903/16/3/73)

## Steps

Work through the following from the top level of this repo.

### 1. Get the dataset

Download the csv folder from the UNSW-NB15 dataset available [here](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Unzip/extract to `data/csv`, the result of `ls data/csv` should include something like:

```
NUSW-NB15_features.csv   
UNSW-NB15_1.csv   
UNSW-NB15_3.csv   
UNSW-NB15_2.csv   
UNSW-NB15_4.csv
...      
```

### 2. Virtual and Vollo environments  

Set up a virtual environment:

```sh
python3 -m venv vollo-venv
source vollo-venv/bin/activate
pip install --upgrade pip
```

Install the Vollo SDK, follow the instructions [here](https://vollo.myrtle.ai/latest/installation.html), this should create a `vollo-sdk-<version>` folder. Now you can install the python dependencies:

```sh
pip install -r requirements.txt
pip install vollo-sdk-*/python/*.whl
```

### 3. Prepare the data

Run the data preparation script:

```sh
python data.py
```

This will take a few minutes and leave its output in `data/processed`.

### 4. Train the model

Run the training script:

```
python train.py
```

This will take several minutes and save the final model in `build/`. In the default configuration the model should have ~ 100k parameters.

## Results

After training our model, we evaluate it on the test set by running:

```
python val.py
```

The model achieved the following results in PyTorch:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 98.9%  |
| Precision | 95.4%  |
| Recall    | 98.6%  |
| F1-score  | 97.0%  |

Note that our results are not directly comparable with those of [^1] since we have trained our model to classify an input window, rather than to classify/predict a forthcoming window following the input window.

You can also run the model in the Vollo VM to check that the results are not negatively affected by Vollo's quantization:

```
python val.py --vollo-vm
```

By default this runs on a small subset of the test set as it will take a long time to run it on the whole test set.

If you have set up a Vollo accelerator with a license, you can run the model on it. You will first need to source the `setup.sh` from the Vollo SDK to set your `LD_LIBRARY_PATH` environment variable so that you can use the Vollo runtime library (`vollo_rt`):

```sh
source vollo-sdk-*/setup.sh
python val.py --vollo
```

Using a Vollo accelerator the latency statistics are:

| Cores | Block size | Mean (us) | Median (us) | P99 (us) |   
|-------|------------|-----------|-------------|----------|
| 1     |       16   |  3.10     |    3.15     |     3.30 |
| 1     |       32   |  2.37     |    2.34     |     2.70 |
 
