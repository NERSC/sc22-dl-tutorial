import torch
import torch.nn as nn

def down_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1),
        nn.LeakyReLU(inplace=True),
    )   

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1, output_padding=0),
        nn.ReLU(inplace=True),
    )


def inverse_transf(x):
    return torch.exp(14.*x)


def loss_func(gen_output, target, params):
    l1_loss = nn.functional.l1_loss(gen_output, target)

    # Transform T and rho back to original space, compute additional L1
    orig_gen = inverse_transf(gen_output[:,0,:,:,:])
    orig_tar = inverse_transf(target[:,0,:,:,:])
    orig_l1_loss = nn.functional.l1_loss(orig_gen, orig_tar)
    return l1_loss + params.LAMBDA_2*orig_l1_loss




class UNet(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.conv_down1 = down_conv(4, 64)
        self.conv_down2 = down_conv(64, 128)
        self.conv_down3 = down_conv(128, 256)
        self.conv_down4 = down_conv(256, 512)        
        self.conv_down5 = down_conv(512, 512)
        self.conv_down6 = down_conv(512, 512)

        self.conv_up6 = up_conv(512, 512)
        self.conv_up5 = up_conv(512+512, 512)
        self.conv_up4 = up_conv(512+512, 256)
        self.conv_up3 = up_conv(256+256, 128)
        self.conv_up2 = up_conv(128+128, 64)
        self.conv_last = nn.ConvTranspose3d(64+64, params.N_out_channels, 4, stride=2, padding=1, output_padding=0)
        
        
    def forward(self, x):
        conv1 = self.conv_down1(x) # 64
        conv2 = self.conv_down2(conv1) # 128
        conv3 = self.conv_down3(conv2) # 256
        conv4 = self.conv_down4(conv3) # 512
        conv5 = self.conv_down5(conv4) # 512
        conv6 = self.conv_down6(conv5) # 512
        
        x = self.conv_up6(conv6) # 512
        x = torch.cat([x, conv5], dim=1)
        x = self.conv_up5(x) # 512
        x = torch.cat([x, conv4], dim=1)
        x = self.conv_up4(x) # 256
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x) # 128
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x) # 64
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_last(x) # 5
        out = nn.Tanh()(x)
        return out

    def get_weights_function(self, params):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, params['conv_scale'])
                if params['conv_bias'] is not None:
                    m.bias.data.fill_(params['conv_bias'])
        return weights_init
 
