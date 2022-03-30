# Stroke prediction

Dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

```py
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the neural network

Edit `src/train.py` for hyperparameters

```py
python3 src/train.py
```

## Use the neural network

Edit `src/predict.py` for the inputs (look at `src/dataset_maps.py` for some mappings)

```py
python3 src/predict.py
```
