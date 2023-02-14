import sys, torch, torchvision, gc
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import numpy as np

class MHeadAttention(nn.Module):
    def __init__(self, n_in, n_out, head_num=4):
        super(MHeadAttention, self).__init__()

        self.head_num = head_num

        self.att = nn.ModuleList([])
        self.cla = nn.ModuleList([])
        for _ in range(self.head_num):
            self.att.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))
            self.cla.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))

        self.head_weight = nn.Parameter(torch.tensor([1.0/self.head_num] * self.head_num))

    def activate(self, x):
        return torch.sigmoid(x)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        x_out = []
        for i in range(self.head_num):
            att = self.att[i](x)
            att = self.activate(att)

            cla = self.cla[i](x)
            cla = self.activate(cla)

            att = att[:, :, :, 0]  # (samples_num, classes_num, time_steps)
            cla = cla[:, :, :, 0]  # (samples_num, classes_num, time_steps)

            epsilon = 1e-7
            att = torch.clamp(att, epsilon, 1. - epsilon)

            norm_att = att / torch.sum(att, dim=2)[:, :, None]
            x_out.append(torch.sum(norm_att * cla, dim=2) * self.head_weight[i])

        x = (torch.stack(x_out, dim=0)).sum(dim=0)

        return x, []

class EffNetAttention(nn.Module):

    def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4, input_seq_length=3000, sampler=None, preserve_ratio=0.1, alpha=1.0, learn_pos_emb=False, n_mel_bins=128, ratio_visualize=1.0, device="cpu"):
        super(EffNetAttention, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.input_seq_length = input_seq_length
        print("Use %s with preserve ratio of %s" % (str(sampler), str(preserve_ratio)))
        self.learn_pos_emb = learn_pos_emb
        self.alpha = alpha
        
        if sampler is None:
            self.neural_sampler = None
            if pretrain == False:
                print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
                self.effnet = EfficientNet.from_name('efficientnet-b'+str(b))
            else:
                print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
                self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b))
        else:
            self.neural_sampler = sampler(input_seq_length, n_mel_bins, 1-preserve_ratio, learn_pos_emb=self.learn_pos_emb)
            if pretrain == False:
                print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
                self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=self.neural_sampler.feature_channels)
            else:
                print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
                self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=self.neural_sampler.feature_channels)
        # multi-head attention pooling
        if head_num > 0:
            print('Model with {:d} attention heads'.format(head_num))
            self.attention = MHeadAttention(
                self.middim[b],
                label_dim)
        else:
            raise ValueError('Attention head must be integer > 0.')
        
        if(n_mel_bins < 128):
            self.avgpool = nn.AdaptiveAvgPool2d((4, 1))
        else:
            self.avgpool = nn.AvgPool2d((4, 1))
            
        self.effnet._fc = nn.Identity()
        self.batch_idx=0
        self.visualize_range = 1200 * ratio_visualize
        self.device = device

    def forward(self, x):
    
        if self.neural_sampler is not None:
            # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
            ret = self.neural_sampler(x)
            x, score = ret['feature'], ret['score']
            # Be careful with this, a too small number car result in a memory leak !
            #if(self.batch_idx == 0 or (self.batch_idx % self.visualize_range == 0 and self.training)):
            #    self.neural_sampler.visualize(ret)
            x = x.transpose(2, 3)
            x = self.effnet.extract_features(x)
            x = self.avgpool(x)
            x = x.transpose(2,3)
        else:
            score = -1
            x = x.unsqueeze(1)
            x = np.tile(x.cpu().numpy(), (1, 3, 1, 1))
            x = torch.from_numpy(x).to(self.device)
            x = self.effnet.extract_features(x)
            x = self.avgpool(x)
            x = x.transpose(2,3)

        out, _ = self.attention(x)
        if(self.training): 
            self.batch_idx += 1

        return out, score
