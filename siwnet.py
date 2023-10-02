import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json

from friction_dataset import FrictionDataset
from loss import trunc_gauss_log_loss

class SIWNet:
    # implementation of the SIWNet model
    def __init__(self, base_net, pi_head, save_path = None, name = None):

        self.base_net = base_net
        self.pi_head = pi_head         
        self.device = torch.device("cuda")
        self.base_net.to(self.device)
        self.pi_head.to(self.device)

        self.save_path = save_path

        if name is None:
            self.name = str(int(time.time()))
        else:
            self.name = name

        self.test_transforms = transforms.Compose([
            transforms.Resize((324, 324)),
            transforms.ToTensor(),
            transforms.Normalize((0.39497562, 0.37916522, 0.35401782), 
                (0.12773657,  0.12749068, 0.12992096))
        ])

        self.params = None
        self.train_transforms = None

    def load_weights(self, path_base, path_pi):
        # function for loading pretrained weights to the model
        self.base_net.load_state_dict(torch.load(path_base))
        self.pi_head.load_state_dict(torch.load(path_pi))
        print("Loaded weights")

    def train(self, params_path, train_path, test_path, train_transforms):
        # function for training the model

        # initialise dataloaders
        self.trainset = FrictionDataset(train_path, train_transforms)
        self.trainloader = torch.utils.data.DataLoader(dataset = self.trainset, 
            batch_size = 32, shuffle = True, num_workers = 4)
        self.testset = FrictionDataset(test_path, self.test_transforms)
        self.testloader = torch.utils.data.DataLoader(dataset = self.testset, 
            batch_size = 32, shuffle = False)
        
        # read parameters from file
        with open(params_path) as params_json:
            self.params = json.load(params_json)
        # store provided image augmentation routine
        self.train_transforms = train_transforms 
        
        # train base net, and then train PI head
        print("Training: " + self.name)
        self._train_base_net()
        self._train_pi_head()

    def _train_base_net(self):
        # function for training the base net

        # check that required initialisations have been made
        if self.params is None or self.train_transforms is None:
            print("Parameters or training image transforms not initialised")
            return
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.base_net.parameters(), lr = self.params["base_lr"], 
                momentum = 0.9, weight_decay = self.params["base_wd"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
        
        # perform training on provided data
        print("Training base net")
        self.base_net.train()
        for epoch in range(self.params["epochs"]):
            if epoch % 10 == 0:
                print("Epoch:", epoch)
            # loop through training data  
            for i, (img, label) in enumerate(self.trainloader):
                optimizer.zero_grad()
                img, label = img.to(self.device), label.to(self.device) 
                # base net outputs the point estimate as well as the features
                prediction, features = self.base_net(img)                 
                label = torch.unsqueeze(label, 1)
                loss = criterion(prediction, label)
                loss.backward()
                optimizer.step()

            scheduler.step()

        # test on provided data
        self.base_net.eval()
        with torch.no_grad():
            # list for saving errors
            test_error = []
            # loop through testing data
            for i, (img, label) in enumerate(self.testloader):
                img, label = img.to(self.device), label.to(self.device) 
                prediction, _ = self.base_net(img)
                label = torch.unsqueeze(label, 1)

                prediction_np = prediction.cpu().numpy()
                label_np = label.cpu().numpy()
                abs_error = np.abs(prediction_np - label_np)
                for j in range(len(abs_error)):
                    test_error.append(abs_error[j])
            
            # compute error metrics from stored errors
            test_error = np.array(test_error)
            mae = np.mean(test_error)
            mse = np.mean(test_error**2)
            rmse = np.sqrt(mse)
            
            # if a path is provided, save the base net weights and result
            print("MAE:", mae, "MSE:", mse, "RMSE:", rmse)
            if self.save_path is None:
                print("Not saving base model, no path provided")
            else:
                # save model
                torch.save(self.base_net.state_dict(), self.save_path + self.name + "_base.pth")
                # save results
                with open(self.save_path + "results_base.csv", 'a') as f:
                    f.write(self.name + "(MAE, MSE, RMSE),")
                    f.write(str(mae) + ',')
                    f.write(str(mse) + ',')
                    f.write(str(rmse) + '\n')

    def _train_pi_head(self):
        # function for training the PI head
        
        # check necessary initialisations
        if self.params is None or self.train_transforms is None:
            print("Parameters or training image transforms not initialised")
            return

        criterion = trunc_gauss_log_loss
        optimizer = optim.SGD(self.pi_head.parameters(), lr = self.params["pi_lr"], 
                momentum = 0.9, weight_decay = self.params["pi_wd"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
        
        # set PI head to training mode and base net to evaluation mode
        self.pi_head.train()
        self.base_net.eval()
        
        # train on provided data
        print("Training PI head")
        for epoch in range(self.params["epochs"]):
            if epoch % 10 == 0:
                print("Epoch:", epoch)
            # loop through training data 
            for i, (img, label) in enumerate(self.trainloader):
                img, label = img.to(self.device), label.to(self.device) 
                # run prediction for friction, disable gradients as base net is frozen
                with torch.no_grad():
                    prediction, features = self.base_net(img)                
     
                optimizer.zero_grad()
                # dropout on features
                features = F.dropout(features, self.params["pi_drp"])
                # concatenate features and point estimate for PI head input
                pi_input = torch.cat((features, prediction), 1)
                # predict standard deviation with PI head
                std = self.pi_head(pi_input)

                label = torch.unsqueeze(label, 1)
                loss = criterion(prediction, std, label)
                loss.backward()
                optimizer.step()

            scheduler.step()
        
        # test on provided data
        self.pi_head.eval()
        self.base_net.eval()
        with torch.no_grad():
            # initialise loss accumulator
            loss_val = 0
            # loop through testing data
            for i, (img, label) in enumerate(self.testloader):
                img, label = img.to(self.device), label.to(self.device) 
                # point estimate and features from base net
                prediction, features = self.base_net(img)
                # concatenate as PI head input
                pi_input = torch.cat((features, prediction), 1)
                # PI head predicts standard deviation
                std = self.pi_head(pi_input)
                
                label = torch.unsqueeze(label, 1) 
                loss = criterion(prediction, std, label)
                loss_val += loss.item()
            
            # if path provided, save model and result
            print("PI loss:", loss_val)
            if self.save_path is None:
                print("Not saving PI head, no path provided")
            else:
                # save model
                torch.save(self.pi_head.state_dict(), self.save_path + self.name + "_pi.pth")
                # save results
                with open(self.save_path + "results_pi.csv", 'a') as f:
                    f.write(self.name + ',')
                    f.write(str(loss_val) + '\n')
    
    def inference(self, img):
        # function for performing inference on provided image

        # base net and PI head to evaluation mode
        self.base_net.eval()
        self.pi_head.eval()
        with torch.no_grad():
            # apply transforms to rescale and normalise image
            img = self.test_transforms(img)
            # add a dimension for poper format (batch size)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)
            # point estimate and features from base net
            prediction, features = self.base_net(img)
            # concatenate to create PI head input
            pi_input = torch.cat((features, prediction), 1)
            # predict standard deviation with PI head
            std = self.pi_head(pi_input)

        return prediction, std

