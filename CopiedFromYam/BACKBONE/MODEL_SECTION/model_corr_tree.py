import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import  load_config
config = load_config()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


def unet_conv(nc, output_nc, kernel_size, outermost=False, innermost=False):
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    downrelu = nn.LeakyReLU(0.2, True)
    # downrelu = nn.ELU(inplace=True)
    downconv = nn.Conv2d(nc, output_nc, kernel_size, stride=2, padding=padding[kernel_size])
    if outermost:
        return nn.Sequential(*[downconv, downrelu])
    elif innermost:
        downnorm = nn.BatchNorm2d(output_nc)
        return nn.Sequential(*[downconv, downnorm, nn.ReLU(True)])
    else:
        downnorm = nn.BatchNorm2d(output_nc)
        return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(nc, output_nc, kernel_size, use_drop=False, outermost=False):
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    uprelu = nn.ReLU(True)
    upconv = nn.ConvTranspose2d(nc, output_nc, kernel_size, stride=2, padding=padding[kernel_size])
    upnorm = nn.BatchNorm2d(output_nc)
    updrop = nn.Dropout(0.5)
    # uprelu = nn.ELU(inplace=True)
    if not outermost:
        if use_drop:
            return nn.Sequential(*[upconv, upnorm, updrop, uprelu])
        else:
            return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Tanh()])


def post_unet(nc):
    postconv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1)
    postnorm = nn.BatchNorm2d(nc)
    postrelu = nn.ReLU(True)
    postconv2 = nn.Conv2d(nc, nc, 3, stride=1, padding=1)
    return nn.Sequential(*[postconv1, postnorm, postrelu, postconv2, nn.Tanh()])


class SplitUNet(nn.Module):
    def __init__(self, ngf=64, nc=1, kernel_size=(2,4), split_location=7):
        super().__init__()
        self.nc = nc
        self.split_location = split_location

        # Encoder layers
        self.encoder = nn.ModuleList([
            unet_conv(nc, ngf, kernel_size, outermost=True),     # convlayer1
            unet_conv(ngf, ngf * 2, kernel_size),               # convlayer2
            unet_conv(ngf * 2, ngf * 4, kernel_size),           # convlayer3
            unet_conv(ngf * 4, ngf * 8, kernel_size),           # convlayer4
            unet_conv(ngf * 8, ngf * 8, kernel_size),           # convlayer5
            unet_conv(ngf * 8, ngf * 8, kernel_size),           # convlayer6
            unet_conv(ngf * 8, ngf * 8, kernel_size),           # convlayer7
            unet_conv(ngf * 8, ngf * 8, kernel_size, innermost=True)  # convlayer8
        ])

        # Decoder layers
        self.decoder = nn.ModuleList([
            unet_upconv(ngf * 8, ngf * 8, kernel_size, use_drop=True),
            unet_upconv(ngf * 16, ngf * 8, kernel_size, use_drop=True),
            unet_upconv(ngf * 16, ngf * 8, kernel_size, use_drop=True),
            unet_upconv(ngf * 16, ngf * 8, kernel_size),
            unet_upconv(ngf * 16, ngf * 4, kernel_size),
            unet_upconv(ngf * 8, ngf * 2, kernel_size),
            unet_upconv(ngf * 4, ngf, kernel_size),
            unet_upconv(ngf * 2, 1, kernel_size, outermost=True)
        ])

        # Shared decoder layers
        self.shared_decoder = nn.ModuleList(self.decoder[:split_location])

        base_head = nn.ModuleList(self.decoder[split_location:])
        self.heads = nn.ModuleList([copy.deepcopy(base_head) for _ in range(2*len(config['Model params']['heads_params'])-1)])

        # Set different random seeds for each head
        for i, head in enumerate(self.heads):
            torch.manual_seed(i)  # Adjust the seed as needed
            head.apply(weights_init)

        self.dim = 1  # Concatenate along the feature dimension for skip connections

    def forward(self, x, mask):
        # Encoder
        features = []
        for conv in self.encoder:
            x = conv(x)
            features.append(x)

        # Bottleneck feature
        bottleneck = features[-1]

        # Shared decoder
        for i, upconv in enumerate(self.shared_decoder):
            if i == 0:
                x = upconv(bottleneck)
            else:
                x = upconv(torch.cat((x, features[-(i + 1)]), self.dim))

        # Separate heads
        # outputs = []
        # for head in self.heads:
        #     head_x = x
        #     for i, upconv in enumerate(head):
        #         head_x = upconv(torch.cat((head_x, features[-(self.split_location + i + 1)]), self.dim)).squeeze()
        #     outputs.append(head_x*mask)

        outputs = []
        mask4 = mask.unsqueeze(1)  # [B,1,H,W] so multiplication is safe

        for head in self.heads:
            head_x = x
            for i, upconv in enumerate(head):
                skip = features[-(self.split_location + i + 1)]
                head_x = upconv(torch.cat((head_x, skip), dim=1))  # keep 4D always

            # now head_x is [B,1,H,W] (because outermost=True ends with 1 channel)
            head_x = head_x.squeeze(1)  # -> [B,H,W] (ONLY remove channel)
            outputs.append(head_x * mask)  # mask is [B,H,W]

        return outputs  # Return all three outputs

# # Test function to compare weights of the heads
# def test_heads_different_weights(model):
#     for i in range(len(model.heads)):
#         for j in range(i + 1, len(model.heads)):
#             for (param1, param2) in zip(model.heads[i].parameters(), model.heads[j].parameters()):
#                 if torch.equal(param1.data, param2.data):
#                     print(f"Head {i} and Head {j} have identical weights {param1.data}.")
#     print("All heads have different weights.")
#     return True


