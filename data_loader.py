import torch
import torch.utils.data
import torchvision
import numpy as np
import pandas as pd

class RegressionTaskData:
    #We need to set the training and testing data loaders
    def __init__(self):
        self.trainloader = self.make_trainloader() #Creating Batches of data for training 
        self.testloader = self.make_testloader() #Creating Batches of data for testing

    def make_trainloader(self):
        train_data = RegressionDataFolder("train") #Loads the train data 
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=256) #creating the batches, so basically when we train the model, it will give the data batch by batch, batch will contains 256 4D tensors, each containing the image, R, t, a
        return trainloader
    
    def make_testloader(self):
        test_data = RegressionDataFolder("test")
        testloader = torch.utils.data.DataLoader(test_data, batch_size=256)
        return testloader

class RegressionDataFolder(torch.utils.data.Dataset):
    def __init__(self, file): 
        #loading data from dataset/train or test.csv location 
        self.data = pd.read_csv(f"dataset/{file}.csv")

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ConvertImageDtype(torch.float)
        ])

    def __len__(self):
        #returns number of samples in data 
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        #Returns image data, pressure and dict of extra values
        return self.transform(torch.from_numpy(np.roll(np.load(self.data.iat[idx, 0]).astype("float32"), np.random.randint(1, 512), axis=0)).unsqueeze(0)), toDNN(torch.from_numpy(np.array(self.data.iat[idx, 4]))), {'R': self.data.iat[idx, 1], 't': self.data.iat[idx, 2], 'a': self.data.iat[idx, 3]}

def toDNN(x):
    return torch.log10(x+1)

def fromDNN(x):
    return torch.pow(10, x) -1

if __name__ == "__main__":
    data = RegressionTaskData()
    print(data.trainloader.dataset.__getitem__(0))
