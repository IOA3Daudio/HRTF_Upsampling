import torch
import torch.nn as nn

class hrtf_net(nn.Module):
    def __init__(self,sparse_num):
        super(hrtf_net, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=sparse_num, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=sparse_num, kernel_size=1),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=440, kernel_size=1),
            nn.BatchNorm2d(440),
            nn.ReLU(),
            

            nn.Conv2d(in_channels=440, out_channels=440, kernel_size=1),
        )

        



    def forward(self, x):
        #(B,ear,pos,freq)
        out_d = self.block_1(x.permute(0,2,1,3))#(B,pos,ear,freq)
        # out_1 = out_d.permute(0,2,1,3)

        out_f = self.block_2(out_d)#(B,freq,ear,pos)
        out_x = out_f.permute(0,2,1,3)

        out = self.block_3(out_d)#(B,freq,ear,pos)
        out = out.permute(0,2,1,3)
        
        return out_x,out
    
    
    
    


















