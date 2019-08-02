import torch
import torch.nn as nn

from config.config import config

class Finetune(nn.Module):
    def __init__(self):
        super(Finetune, self).__init__()
        self.extralayers = nn.Sequential(
            nn.Conv2d(1024, 4096, 1, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace = True),
            nn.Conv2d(4096, 4096, 1, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace = True),
            nn.Conv2d(4096, 4096, 1, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace = True),
            nn.Conv2d(4096, 2048, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace = True),
            nn.Conv2d(2048, 1024, 1, 1),
            )
        self.MLP = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Conv2d(1024, 512, 1, 1),
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,inputs):
        inputs = self.extralayers(inputs) # b*1024*c*r
        batch, channel, col, row = inputs.shape
        inputs = nn.MaxPool2d((col,row))(inputs) #b*1024*1*1
        inputs = self.MLP(inputs)

        return inputs

    def load_dict(self,model, cuda = True):
        if cuda:
            cudnn.benchmark = True
            model.load_state_dict(torch.load(config['resume']))
            model = model.cuda()
        else:
            self.sst.load_state_dict(torch.load(config['resume'], map_location='cpu'))
        model.eval()


