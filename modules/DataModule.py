import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class PUFDatasetFourClass(Dataset):
    def __init__(self, file, timestamp, cbits, ids):
        self.data_file = file
        # self.data.columns.get_loc('2.034068136272545e-08') -> 203
        self.timestamp = int(timestamp)
        self.cbits = cbits
        self.data = pd.read_csv(self.data_file).iloc[ids]
        # XOR case - we already have selected the timestamp
        if len(self.data.columns) == 3:
            self.data = self.data[['challenge', 'r']]
            self.timestamp = 'r'
        # non-XOR case - we need to select the timestamp manually
        else:
            self.data.columns = ['challenge'] + list(
                range(1, len(self.data.columns)))
        self.data = self.data.sample(frac=1, random_state=17)

    # 2.034068136272545e-08

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        challenge = list(int(x) for x in
                         ("{0:0" + str(self.cbits) + "b}").format(
                             int(item['challenge'])))
        response = item[self.timestamp]
        if response == 0:
            response = torch.tensor([0])
        elif response == 1:
            response = torch.tensor([1])
        elif response == 10:
            response = torch.tensor([2])
        elif response == 11:
            response = torch.tensor([3])
        else:
            print("ERRR")
        return torch.tensor(challenge, dtype=torch.float), response


class PUFDataModule(LightningDataModule):
    def __init__(self, batch_size, file, timestamp, cbits, train_ids, val_ids, test_ids):
        super().__init__()
        self.batch_size = batch_size
        self.file = file
        self.timestamp = str(timestamp)
        self.cbits = cbits
        self.train_kwargs = {"batch_size": self.batch_size, "num_workers": 4,
                             "pin_memory": True, "shuffle": True}
        self.val_test_kwargs = {"batch_size": self.batch_size, "num_workers": 4,
                                "pin_memory": True}
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

    def setup(self, stage=None):
        self.train_dataset = PUFDatasetFourClass(self.file, self.timestamp, self.cbits,
                                                 self.train_ids)
        self.val_dataset = PUFDatasetFourClass(self.file, self.timestamp, self.cbits,
                                               self.val_ids)
        self.test_dataset = PUFDatasetFourClass(self.file, self.timestamp, self.cbits,
                                                self.test_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_test_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)


'''class PUFDatasetMultiClass(Dataset):
    def __init__(self, file, cbits, ids):
        self.data_file = file
        self.timestamp = 100
        self.cbits = cbits
        self.data = pd.read_csv(self.data_file).iloc[ids]
        self.data.columns = ['challenge'] + list(range(1, 500))
        self.data = self.data[['challenge', self.timestamp]]
        self.data = self.data.sample(frac=1, random_state=17)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        challenge = list(int(x) for x in
                         ("{0:0" + self.cbits + "b}").format(item['challenge']))
        response = item[self.timestamp]
        if response < 10:
            response = torch.tensor([0, response], dtype=torch.float)
        else:
            response = torch.tensor([int(x) for x in str(response)],
                                    dtype=torch.float)
        return torch.tensor(challenge, dtype=torch.float), response'''
