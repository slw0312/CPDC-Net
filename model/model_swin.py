from torchvision.models.video import swin_transformer
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
import torch
from torch import nn
from decoder1 import *

# 使用swin3d_t函数创建Video Swin Transformer模型
# model=swin_transformer.SwinTransformer3d()
model = swin_transformer.SwinTransformer3d(patch_size=[2, 4, 4], embed_dim=96, depths=[2, 2, 6, 2],
                                           num_heads=[3, 6, 12, 24], window_size=[8, 7, 7])
checkpoint = torch.load("swin3d_t-7615ae03.pth")
model.load_state_dict((checkpoint))

#
# # x = torch.randn(2, 3, 10, 224, 224)
#
#
class video_swin_decoder(nn.Module):
    def __init__(self):
        super(video_swin_decoder, self).__init__()
        self.swin = model
        self.decoder = Decoder()
        self.fc=nn.Linear(768,512)
        self.fc5=nn.Linear(245,10)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        # self.swin = Swin_Model()
        # self.decoder=Decoder()

    def forward(self, input, caption, speed, course, goal, accel):
        x = self.swin.patch_embed(input)
        x = self.swin.pos_drop(x)
        x = self.swin.features(x)
        x = self.swin.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = torch.flatten(x, 2).permute(0, 2, 1)
        #print(x.shape)
        x=self.fc(x)
        x=x.permute(0,2,1)
        x=self.fc5(x)
        #x=x.squeeze()
        x=x.permute(0,2,1)
        speed = self.fc1(speed)
        course = self.fc2(course)
        goal = self.fc3(goal)
        accel = self.fc4(accel)
        x = torch.cat((x, speed), dim=1)
        x = torch.cat((x, course), dim=1)
        x = torch.cat((x, goal), dim=1)
        x = torch.cat((x, accel), dim=1)

        # print(x.shape)
        x = self.decoder(caption, x)
        # print(x.shape)

        return x
#*************************
