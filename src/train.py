import os
import pandas as pd
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import StrokeDataset


DATASET_SPLIT = 0.9 # n% training data, 1 - n% testing data

LEARNING_RATE = 0.1
BATCH_SIZE = 64
EPOCHS = 50
NORMALIZE = False

current_directory = os.path.dirname(os.path.realpath(__file__))

start_time = time.time()
model_folder_path = os.path.realpath(f'{current_directory}/../models/model_{start_time:.4f}/')

# Load dataset

dataset_path = current_directory + '/../dataset/dataset.csv'
dataset = StrokeDataset(
    dataset_path,
    normalize=NORMALIZE,
    # convert pandas object to tensor
    feature_transform=lambda feature: torch.tensor(feature, dtype=torch.float32),
    label_transform=lambda label: torch.reshape(torch.tensor(label, dtype=torch.float32), (-1,))
)

split_pos = int(len(dataset) * DATASET_SPLIT)
train_dataset, test_dataset = random_split(dataset, [split_pos, len(dataset) - split_pos])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Build model

while (load_existing_model := input('Load existing model? (Y/n) ')).lower() not in ['y', 'n']:
    pass

if load_existing_model == 'y':
    while True:
        existing_model_path = input('Existing model path: ')
        try:
            model = torch.load(existing_model_path)
            break
        except:
            print('Invalid model file')

else:
    input_size = len(dataset[0][0])

    model = nn.Sequential(
        nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

print(f'Model: {model}')

# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# Make training and testing functions

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def train_loop():
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE
            print(f'Loss: {loss:>7f} [{current:>4d}/{size:>4d}]')


def test_loop():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            pred_answers = (pred > 0.5).float() # 0 or 1 only
            correct += torch.sum(torch.logical_not(torch.logical_xor(pred_answers, y))).item()

    avg_loss = test_loss / num_batches
    accuracy = correct / size
    print(f'Accuracy: {100 * accuracy:>0.1f}%, Average loss: {avg_loss:>8f}')


# Train and test every epoch

try:
    for epoch in range(EPOCHS):
        print()
        print(f'Epoch {epoch + 1}')
        print('-' * 20)
        train_loop()
        test_loop()

        if not os.path.isdir(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(model, f'{model_folder_path}/epoch_{epoch + 1}.pt')

    print('Finished')

except KeyboardInterrupt:
    print('Finished early')
