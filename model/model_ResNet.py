import torchvision.models as models


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
from resnet import *
# clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
# clip_model=clip_model.train()
net = ResNet50()
class resnet_decoder(nn.Module):
    def __init__(self):
        super(resnet_decoder, self).__init__()
        #self.clip=clip_model
        #self.swin = model
        # self.text_encoder=caption_encoder()

        self.encoder = net
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.encoder1 = Encoder()
        # self.encoder2 = Encoder()
        self.decoder1= Decoder()
        #self.decoder2 = Decoder(vocab=12)
        #self.decoder2 = Decoder()
        #self.decoder2 = Decoder()
        # self.decoder3 = Decoder()
        # self.decoder4 = Decoder()
        self.fc = nn.Linear(1024, 512)
        # self.fc1=nn.Linear(512,512)
        # self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(512, 512)
        # self.fc4 = nn.Linear(512, 512)
        #self.fc5 = nn.Linear(14, 1)

    def forward(self, input, caption, speed, course, goal, acc):
        x=self.encoder(input)
        x=self.avgpool(x)
        # print(x.shape)
        x=x.view(x.size(0), 1,-1)
        x=self.fc(x)
        # print(x.shape)
        # x=x.reshape(-1,10,1000)
        # x=self.fc(x)
        #
        # speed=self.fc1(speed)
        #
        #
        # course = self.fc2(course)
        # goal = self.fc3(goal)
        # acc = self.fc4(acc)
        # x = torch.cat((x, speed), dim=1)
        # x = torch.cat((x, course), dim=1)
        # x = torch.cat((x, goal), dim=1)
        # x = torch.cat((x, acc), dim=1)
        # x=self.encoder1(x)
        #
        # x = self.encoder2(x)


        y1 = self.decoder1(caption, x)




        return y1
