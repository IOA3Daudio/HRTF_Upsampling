import torch
import torch.nn as nn

class itd_net(nn.Module):
    def __init__(self):
        super(itd_net, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=121, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(in_channels=512, out_channels=2048, kernel_size=1),
        )

        self.block_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=121, kernel_size=1),
        )

        self.block_3 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=440, kernel_size=1),
            nn.BatchNorm1d(440),
            nn.ReLU(),

            nn.Conv1d(in_channels=440, out_channels=440, kernel_size=1),
        )

        



    def forward(self, x):
        out_d = self.block_1(x.permute(0,2,1))
        out_f = self.block_2(out_d)
        out_x = out_f.permute(0,2,1)
        out = self.block_3(out_d)
        out = out.permute(0,2,1)
        return out_x,out
    
    
    
    


















