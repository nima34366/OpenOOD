# import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# class Binary_Classifier(pl.LightningModule):
#     def __init__(self, n_inputs, n_outputs):
#         super().__init__()
#         self.linear = torch.nn.Linear(n_inputs, n_outputs)
#     def forward(self, x):
#         y_pred = torch.sigmoid(self.linear(x))
#         return y_pred
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self.forward(x)
#         loss = criterion(logits, y)
#         self.log("train_loss", loss)
#         return loss
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self.forward(x)
#         loss = criterion(logits, y)
#         self.log("val_loss", loss)
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
#         return optimizer
    
# class Secondary_DataModule(pl.LightningDataModule):
#     def __init__(self, batch_size: int = 32):
#         super().__init__()
#         self.batch_size = batch_size
#     def setup(self, stage):
#         self.dataset_train = Secondary_Dataset()
#         self.dataset_test = Secondary_Dataset(train=False)
#     def train_dataloader(self):
#         return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=20)
#     def val_dataloader(self):
#         return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=20)
class Secondary_Dataset(Dataset):
    def __init__(self, dirx='/home/nima/OpenOOD/scripts/classifier_2/data/x.npy', diry='/home/nima/OpenOOD/scripts/classifier_2/data/y.npy', train = True, test_size = 0.2, random_state = 42):
        self.x = np.load(dirx)
        self.y = np.load(diry)
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size, random_state=self.random_state)  
        self.X_train, self.X_test, self.y_train, self.y_test = torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32).view(self.y_train.shape[0],1), torch.tensor(self.y_test, dtype=torch.float32).view(self.y_test.shape[0],1)
    def __len__(self):
        if self.train==True:
            return len(self.y_train)
        else:
            return len(self.y_test)
    def __getitem__(self, idx):
        if self.train == True:
            return self.X_train[idx][[0,9]], self.y_train[idx]
        else:
            return self.X_test[idx][[0,9]], self.y_test[idx]

train_dataloader = DataLoader(Secondary_Dataset(), batch_size=16, shuffle=True)
val_dataloader = DataLoader(Secondary_Dataset(train=False), batch_size=16, shuffle = False)

# X_train, X_test, y_train, y_test = train_test_split(np.load('/home/nima/OpenOOD/scripts/classifier_2/data/x.npy'), np.load('/home/nima/OpenOOD/scripts/classifier_2/data/y.npy'), test_size=0.2, random_state=42)

class Net(torch.nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = torch.nn.Linear(input_shape,10)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10,5)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(5,1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = torch.sigmoid(self.fc3(x))
        return x
  
model = Net(2)
epochs = 1000

optimizer = torch.optim.SGD(model.parameters(),lr=0.00001)

criterion = torch.nn.BCELoss()

Loss = []
acc = []
itr = 0
number_of_epochs=100
for epoch in range(number_of_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        y_prediction=model(images)
        loss=criterion(y_prediction,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    Loss.append(loss.item())
    correct = 0
    for images, labels in val_dataloader:
        outputs = model(images)
        predicted = torch.round(outputs)
        correct += (predicted.eq(labels).sum())/len(labels)
    accuracy = 100 * (correct.item()) / len(val_dataloader)
    acc.append(accuracy)
    print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))




        
# data_module = Secondary_DataModule()

# model = Binary_Classifier(13,2)
# trainer = pl.Trainer()
# trainer.fit(model, data_module)
    
