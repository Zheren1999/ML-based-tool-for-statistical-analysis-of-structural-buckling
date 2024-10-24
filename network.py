import torch
import torch.nn as nn
import numpy as np
import sys
#import matplotlib.pyplot as plt
from data_loader import *

def main():
    #Chooses which device to use
    #if GPU is available uses it, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set is as the default device
    torch.set_default_device(device)
    print(F"Using the device: {device}")
    #Loading the training and testing data 

    regression_task = RegressionTaskData() 

    #Train model
    epochs = 1000
    model = train_network(device, epochs, regression_task)

    #Save model
    try:
        filename = f"shells_{sys.argv[1]}.pth"
    except:
        filename = "shells_" + str(int(torch.rand(1) * 10000)).zfill(4) + ".pth"
    save_model(model, filename)

    #Load model
    model = load_model(filename)
    model.to(device)

    #Evaluate model
    print(f"{evaluate_network(model, device, regression_task):.4f}")

#function that is doing the training of the model
def train_network(device, n_epochs, regression_task):
    #Define the model
    model = CNNRegression() #Create an instance of CNN Regression class 
    model.to(device) #Move the model to the specified device (GPU if available or CPU)
    print(model)
    criterion = nn.MSELoss() #The seme as the loss function, we basically defining the criterion for model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #Adam optimizer 

    for epoch in range(n_epochs): #Goes through dataset n_epochs times
        for i, (inputs, targets, extra) in enumerate(regression_task.trainloader): #This inner loop goes through batches of data provided by regression_task.trainloader
            #Input is the image, target is the pressure, extra - dictionary containing extra parameters (R, t, a)
            #Zero gradients, clear the gradient, that they will not accumulate
            optimizer.zero_grad()

            #Forward
            outputs = model(inputs.to(device), {'R': extra['R'], 't': extra['t'], 'a': extra['a']}).flatten()
            #y_value output from the model, which takes image, and values of R, t, a
            loss = criterion(outputs, targets.type(torch.FloatTensor).to(device))

            #Backward & optim           
            loss.backward()
            optimizer.step()

        #Print stats
        print(F"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}, Validation loss: {evaluate_network(model, device, regression_task):.4f}")
    
    return model

def save_model(model, filename):
    #Save the model, basically saving the parameters og the model 
    torch.save(model.state_dict(), filename)

def load_model(filename):
    #Load the model
    #Create new instance of the model
    #Direct our saved model parameters into the new instance of the CNNRegression Class
    model = CNNRegression()
    model.load_state_dict(torch.load(filename))
    return model

#Testing function, we basically divided training and testing into 2 loops 
def evaluate_network(model, device, regression_task):
    #Use the same criterion that we used before MSE loss
    criterion = nn.MSELoss()
    #The same as torch.inference_mode context manager, just running evalution without running any oprimization (no gradients)
    #Only evalutiong model, to training it 

    with torch.no_grad():
        outputs_list, targets_list = [],[]
        #model predicted value and model true value 

        total_loss = 0
        n_samples_total = 0
        #Go through each batch in the test data 
        #In testing and training we go through batches because the dataset is really big 


        for i, (inputs, targets, extra) in enumerate(regression_task.testloader):
            #Forward pass
            outputs = model(inputs.to(device), {'R': extra['R'], 't': extra['t'], 'a': extra['a']}).flatten()
            #Morgan appended values not normally, using the following formula 10^x-1. I do not why 

            outputs_list.extend(fromDNN(outputs))
            targets_list.extend(fromDNN(targets))

            loss = criterion(outputs, targets.to(device))
            total_loss += loss.item()

            n_samples_total += len(outputs)

        mean_loss = total_loss / len(regression_task.testloader)

    # plt.hist(np.array(outputs_list) - np.array(targets_list))
    # plt.show()

    # plt.scatter(outputs_list, targets_list)
    # plt.xlabel("Predicted Pressures")
    # plt.ylabel("Actual Pressures")
    # plt.show()

    return(mean_loss)


class CNNRegression(nn.Module):
    #In out particular task we use the CNN Regression model to predict the buckling pressure involving images representation
    
    def __init__(self):
        super(CNNRegression, self).__init__()
        #Input size is 1 picture with 256X256 pixels


        #Convolutional layers: filters might detect edges, textures, specific shapes)
        #This layer expects 2Dimage with 1 channel = in_channels, outer_channel is 16 features (detecting different channels)
        #The size of the kernel is 3X3 (fiter (kernel)), filter eill move 3 by 3 matrix to detect the features
        #Basically we are doing convolution process to do edge detection
        #We are multiplying the filter function (that contains 9 elements) with the each pixels (contains red, blue, green color values) to get new value (useful for recognizing the edges)
        #Now we will have new 'color' for each pixels on the new picture (kind of averaging the values)
        #stride controls how much the filter moves by 1 pixel at a time 
        #padding controls adding extra pixels around the picture (1 layer of zeros will be added around the boarder of the input)

        #Output size is 16 channel with 256X256 feature map
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        #Pooling is reducing the computational complexity (also model becomes less sensetive to small shifts in the input)
        #Going with the matrix 2X2 and find the maximum of the value 
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #Output size is 16 channels with 128X128 feature map 

        #Another one convolutional layer, tales 16 features and produces 64 features with more complex features
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        #Output size is 64 channels with 128X128 feature map 

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #Output size is 64 channels wuth 64X64 feature map 

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        #Output size is 128 channels with 64X64 feature map 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #Output size is 128 channels with 32X32 feature map 
        self.gap = nn.AdaptiveAvgPool2d((1, 128))
        #Adaptive Average Pooling works like a regular pooling but it apadts the output  with a apecified size, so in our case the height = 1 and width = 128

        #Output size = 128 channels with the size of 1X128 feature map 

        #self.linear_line_size = int(128*64*64)

        #before passing putput from convolutional layers to fully conncected layers it is flattened into 1 d vector 128X1X128=16384 features
        
        
        #self.fc1 = nn.Linear(in_features=4096, out_features=128)
        #self.fc2 = nn.Linear(in_features=131, out_features=1)

        self.fc1 = nn.Linear(in_features=16384, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
    
    def forward(self, x, add): #x in this case is the image (matrix representation), add - dictionary with the parameters R, t, a
        x = self.conv1(x)
        # print('Size of tensor after each layer')
        # print(f'conv1 {x.size()}')
        x = nn.functional.relu(x)

        #ReLu function introducing non-linearity into the model. Without non-linearity a NN consisting only of linear operators will act like large linear model 
        #nomatter how many layers we add, the entire network will be equilvalent just 1 big linear transformation which is not very powerful
        #ReLU allowing a network to learn and represnt more complex patters 
        #ReLu (x)- max(0, x) this function activates cetran neurons and supresses other, allowing the network to focus on important features while ignoring less important
    

        # print(f'relu1 {x.size()}')
        x = self.pool1(x)
        # print(f'pool1 {x.size()}')
        x = self.conv2(x)
        # print(f'conv2 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.pool2(x)
        # print(f'pool2 {x.size()}')

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.gap(x)

        #x = x.view(-1, self.linear_line_size)
        # print(f'view1 {x.size()}')
        x = self.fc1(x)
        # print(f'fc1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.fc2(torch.hstack((x, torch.hstack((add['R'].view(-1, 1), add['t'].view(-1, 1), add['a'].view(-1, 1))))).float())
        # print(f'fc2 {x.size()}')
        return x.float()

if __name__ == "__main__":
    main()
