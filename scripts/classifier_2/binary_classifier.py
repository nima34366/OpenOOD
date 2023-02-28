import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

criterion = torch.nn.CrossEntropyLoss()

class Binary_Classifier(pl.LightningModule):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = criterion(logits, y)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = criterion(logits, y)
        self.log("val_loss", loss)
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer
    
class Secondary_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
    def setup(self, stage):
        self.dataset_train = Secondary_Dataset()
        self.dataset_test = Secondary_Dataset(train=False)
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=20)
    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=20)

class Secondary_Dataset(Dataset):
    def __init__(self, dirx='/home/nima/OpenOOD/scripts/classifier_2/x.npy', diry='/home/nima/OpenOOD/scripts/classifier_2/y.npy', train = True, test_size = 0.2, random_state = 42):
        self.x = np.load(dirx)
        self.y = np.load(diry)
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size, random_state=self.random_state)  

    def __len__(self):
        if self.train==True:
            return len(self.y_train)
        else:
            return len(self.y_test)

    def __getitem__(self, idx):
        if self.train == True:
            return self.X_train[idx].astype('float32'), self.y_train[idx]
        else:
            return self.X_test[idx].astype('float32'), self.y_test[idx]
        
data_module = Secondary_DataModule()

model = Binary_Classifier(13,2)
trainer = pl.Trainer(accelerator='gpu', devices=1)
trainer.fit(model, data_module)
    
