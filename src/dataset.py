import pandas as pd
from torch.utils.data import Dataset

import dataset_maps


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class StrokeDataset(Dataset):
    def __init__(self, dataset_file_path, normalize=True, feature_transform=None, label_transform=None):
        self.feature_transform = feature_transform
        self.label_transform = label_transform

        self.df = pd.read_csv(dataset_file_path)

        # map strings to numbers
        self.df['gender'] = self.df['gender'].map(dataset_maps.gender_map, na_action='ignore')
        self.df['ever_married'] = self.df['ever_married'].map(dataset_maps.ever_married_map, na_action='ignore')
        self.df['work_type'] = self.df['work_type'].map(dataset_maps.work_type_map, na_action='ignore')
        self.df['Residence_type'] = self.df['Residence_type'].map(dataset_maps.residence_type_map, na_action='ignore')
        self.df['smoking_status'] = self.df['smoking_status'].map(dataset_maps.smoking_status_map, na_action='ignore')

        # remove rows with missing data
        self.df.dropna(inplace=True)

        # normalize
        if normalize:
            for col in self.df.columns:
                self.df[col] = self.df[col] / self.df[col].abs().max()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # https://stackoverflow.com/a/29763653
        features = self.df.iloc[index].drop(['stroke', 'id'])
        label = self.df['stroke'].iloc[index]

        if self.feature_transform:
            features = self.feature_transform(features)
        if self.label_transform:
            label = self.label_transform(label)

        return features, label
