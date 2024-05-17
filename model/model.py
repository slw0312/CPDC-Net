from VIT import *
from swin import *
# from decoder import *
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=False)
#
# # 去除最后一层
# model = nn.Sequential(*list(model.children())[:-1])
#
#
# class vit_decoder(nn.Module):
#     def __init__(self):
#         super(vit_decoder, self).__init__()
#         self.vit = model
#         self.decoder = Decoder()
#         # self.swin = Swin_Model()
#
#     def forward(self, input, caption, speed, course, goal):
#         x = self.vit(input)
#         x = x.reshape(-1, 10, 196, 768)
#         x = x.reshape(-1, 1960, 768)
#         x = torch.cat((x, speed), dim=1)
#         x = torch.cat((x, course), dim=1)
#         x = torch.cat((x, goal), dim=1)
#
#         # print(x.shape)
#         x = self.decoder(caption, x)
#         # print(x.shape)
#
#         return x

#*************************************************************************
from torchvision.models.video import swin_transformer
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
import torch
from torch import nn
# from decoder import *

# 使用swin3d_t函数创建Video Swin Transformer模型
# model=swin_transformer.SwinTransformer3d()
#model = swin_transformer.SwinTransformer3d(patch_size=[2, 4, 4], embed_dim=96, depths=[2, 2, 6, 2],
#                                            num_heads=[3, 6, 12, 24], window_size=[8, 7, 7])
# #checkpoint = torch.load("swin3d_t-7615ae03.pth")
# #model.load_state_dict((checkpoint))
#
#
# # x = torch.randn(2, 3, 10, 224, 224)
#
#
# class video_swin_decoder(nn.Module):
#     def __init__(self):
#         super(video_swin_decoder, self).__init__()
#         self.swin = model
#         self.decoder = Decoder()
#         self.fc1 = nn.Linear(768, 768)
#         self.fc2 = nn.Linear(768, 768)
#         self.fc3 = nn.Linear(768, 768)
#         self.fc4 = nn.Linear(768, 768)
#         # self.swin = Swin_Model()
#         # self.decoder=Decoder()
#
#     def forward(self, input, caption, speed, course, goal, accel):
#         x = self.swin.patch_embed(input)
#         x = self.swin.pos_drop(x)
#         x = self.swin.features(x)
#         x = self.swin.norm(x)
#         x = x.permute(0, 4, 1, 2, 3)
#         x = torch.flatten(x, 2).permute(0, 2, 1)
#         speed = self.fc1(speed)
#         course = self.fc2(course)
#         goal = self.fc3(goal)
#         accel = self.fc4(accel)
#         x = torch.cat((x, speed), dim=1)
#         x = torch.cat((x, course), dim=1)
#         x = torch.cat((x, goal), dim=1)
#         x = torch.cat((x, accel), dim=1)
#
#         # print(x.shape)
#         x = self.decoder(caption, x)
#         # print(x.shape)
#
#         return x
#************************************************************************
from VIT import *
from swin import *
from decoder1 import *
import torch
from Transformer import *
import clip
#from caption_encoder import *
# model = swin_transformer.SwinTransformer3d(patch_size=[2, 4, 4], embed_dim=96, depths=[2, 2, 6, 2],
#                                            num_heads=[3, 6, 12, 24], window_size=[8, 7, 7])
# checkpoint = torch.load("swin3d_t-7615ae03.pth")
# model.load_state_dict((checkpoint))

# clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
# clip_model=clip_model.train()
class clip_decoder(nn.Module):
    def __init__(self):
        super(clip_decoder, self).__init__()
        #self.clip=clip_model
        #self.swin = model
        # self.text_encoder=caption_encoder()
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.decoder1= Decoder()
        #self.decoder2 = Decoder(vocab=12)
        #self.decoder2 = Decoder()
        #self.decoder2 = Decoder()
        # self.decoder3 = Decoder()
        # self.decoder4 = Decoder()

        self.fc1=nn.Linear(10,512)
        self.fc2 = nn.Linear(10, 512)
        self.fc3 = nn.Linear(10, 512)
        self.fc4 = nn.Linear(10, 512)
        #self.fc5 = nn.Linear(14, 1)

    def forward(self, input, caption, speed, course, goal, acc):

        speed=self.fc1(speed)


        course = self.fc2(course)
        goal = self.fc3(goal)
        acc = self.fc4(acc)
        # x = torch.cat((input, speed), dim=1)
        x = torch.cat((input, course), dim=1)
        x = torch.cat((x, goal), dim=1)
        x = torch.cat((x, acc), dim=1)
        x=self.encoder1(x)

        x = self.encoder2(x)
        # print(x.shape)


        y1 = self.decoder1(caption, x)




        return y1
