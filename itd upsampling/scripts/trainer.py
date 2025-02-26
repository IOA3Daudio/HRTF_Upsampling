import torch
import torch.nn as nn
import numpy as np
import os

class trainer(object):
    def __init__(self,configs,fold,device,train_dataloader,valid_dataloader,optimizer,lossF):
        seed = configs.seed
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.lossF = lossF
        self.model_save_pth = configs.model_save_pth+str(fold)
        if not os.path.exists(self.model_save_pth):
            os.makedirs(self.model_save_pth)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=configs.scheduler_step_size,gamma = configs.scheduler_gamma)

    def train(self,net,epoch,history):
        net.train(True)
        train_loss = 0
        dataloader = self.train_dataloader
        for batch_idx, (x_data,y_data) in enumerate(dataloader):
            x_data = x_data.to(self.device)
            y_data = y_data.to(self.device)
            self.optimizer.zero_grad()
            out_x,out = net(x_data)
            loss_1 = self.lossF(out_x, x_data)
            loss_2 = self.lossF(out, y_data)
            loss = loss_1 + loss_2
            
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                train_loss += loss.item()
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                        epoch, batch_idx, len(dataloader),100. * batch_idx / len(dataloader), loss.item()))
        net.train(False)
        with torch.no_grad():
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader)))
            history['Train Loss'].append(train_loss / len(dataloader))
            # if epoch >= int(EPOCHS*3/4) - 1:
            torch.save(net.state_dict(),self.model_save_pth+'/net'+str(epoch)+'.pt')

        return history
    
    def valid(self,net,history):
        net.train(False)
        valid_loss = 0
        dataloader = self.valid_dataloader
        with torch.no_grad():
            for (x_data,y_data) in dataloader:
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                out_x,out = net(x_data)
                loss_1 = self.lossF(out_x, x_data)
                loss_2 = self.lossF(out, y_data)
                loss = loss_1 + loss_2

                valid_loss += loss.item()
            print('====> Valid Average loss: {:.4f}'.format(valid_loss / len(dataloader)))  
            history['Valid Loss'].append(valid_loss / len(dataloader))

        return history
