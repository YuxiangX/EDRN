from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return EDRN(args)

class EDRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDRN, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = 64
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [EDRM(conv)for _ in range(1)]
        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res = res + x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
'''..................................................................................................'''
class EDRM(nn.Module):
    def __init__(self,conv):
        super(GDN, self).__init__()
        n_feat = 64
        n_splite = 8
        self.distill_learning_1 = Conv_Group(conv,n_feat_b=n_feat, n_feat_a=n_feat, n_residual=4)
        self.distill_learning_2 = Conv_Group(conv,n_feat_b=n_feat - n_splite * 1, n_feat_a=n_feat - n_splite * 1, n_residual=4)
        self.distill_learning_3 = Conv_Group(conv,n_feat_b=n_feat - n_splite * 2, n_feat_a=n_feat - n_splite * 2, n_residual=4)
        self.distill_learning_4 = Conv_Group(conv,n_feat_b=n_feat - n_splite * 3, n_feat_a=n_feat - n_splite * 3, n_residual=4)
        self.distill_learning_5 = Conv_Group(conv,n_feat_b=n_feat - n_splite * 4, n_feat_a=n_feat - n_splite * 4, n_residual=4)
        self.conv1 = conv(n_feat,n_feat, 1)
        self.conv = conv(n_feat, n_feat, 3)
        self.n_splite = n_splite
    def forward(self, x):
        _,C,_,_ = x.shape
        y = self.distill_learning_1(x)
        y, dr_1 = torch.split(y, (C-self.n_splite * 1, self.n_splite* 1), dim=1)
        y = self.distill_learning_2(y)
        y, dr_2 = torch.split(y, (C-self.n_splite * 2, self.n_splite* 1), dim=1)
        y = self.distill_learning_3(y)
        y, dr_3 = torch.split(y, (C-self.n_splite * 3, self.n_splite* 1), dim=1)
        y = self.distill_learning_4(y)
        y, dr_4 = torch.split(y, (C-self.n_splite * 4, self.n_splite* 1), dim=1)
        y = self.distill_learning_5(y)
        output  = torch.cat((y,dr_1,dr_2,dr_3,dr_4),dim=1)
        output  = self.conv1(output)
        output = self.conv(output)
        return output

class Conv_Group(nn.Module):
    def __init__(
            self, conv, n_feat_b =64, n_feat_a = 64, conv1_h = True, conv1_t = False, n_residual = 4):
        super(Conv_Group, self).__init__()
        m_body = []
        if conv1_h :
            m_body.append(conv(n_feat_b,n_feat_a,1))
        for _ in range(n_residual):
            m_body.append(common.ResBlock(conv,n_feat_a,kernel_size=3,act=nn.ReLU(True),res_scale=1))
        self.body = nn.Sequential(*m_body)
    def forward(self, x_body):
        y = self.body(x_body)
        return y

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        m.append(conv(n_feats,3*scale**2,3,bias))
        m.append(nn.PixelShuffle(scale))
        super(Upsampler, self).__init__(*m)
