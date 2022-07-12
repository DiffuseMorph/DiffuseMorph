import torch
import torch.nn as nn


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Networks
##############################################################################

# Spatial Transformer 3D Net #################################################
class Dense2DSpatialTransformer(nn.Module):
    def __init__(self):
        super(Dense2DSpatialTransformer, self).__init__()

    def forward(self, input1, input2):
        return self._transform(input1, input2[:, 0], input2[:, 1])

    def _transform(self, input1, dHeight, dWidth):
        batchSize = dHeight.shape[0]
        hgt = dHeight.shape[1]
        wdt = dHeight.shape[2]

        H_mesh, W_mesh = self._meshgrid(hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, hgt, wdt)
        H_upmesh = dHeight + H_mesh
        W_upmesh = dWidth + W_mesh

        return self._interpolate(input1, H_upmesh, W_upmesh)

    def _meshgrid(self, hgt, wdt):
        h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()
        w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()
        return h_t, w_t

    def _interpolate(self, input, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch    = input.shape[1]
        height = input.shape[2]
        width  = input.shape[3]

        img = torch.zeros(nbatch, nch, height+2, width+2).cuda()
        img[:, :, 1:-1, 1:-1] = input
        img[:, :, 0, 1:-1] = input[:, :, 0, :]
        img[:, :, -1, 1:-1] = input[:, :, -1, :]
        img[:, :, 1:-1, 0] = input[:, :, :, 0]
        img[:, :, 1:-1, -1] = input[:, :, :, -1]
        img[:, :, 0, 0] = input[:, :, 0, 0]
        img[:, :, 0, -1] = input[:, :, 0, -1]
        img[:, :, -1, 0] = input[:, :, -1, 0]
        img[:, :, -1, -1] = input[:, :,-1, -1]

        imgHgt = img.shape[2]
        imgWdt = img.shape[3]

        # H_upmesh, W_upmesh = [H, W] -> [BHW,]
        H_upmesh = H_upmesh.view(-1).float()+1.0  # (BHW,)
        W_upmesh = W_upmesh.view(-1).float()+1.0  # (BHW,)

        # H_upmesh, W_upmesh -> Clamping
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        hf = torch.clamp(hf, 0, imgHgt-1)  # (BHW,)
        hc = torch.clamp(hc, 0, imgHgt-1)  # (BHW,)
        wf = torch.clamp(wf, 0, imgWdt-1)  # (BHW,)
        wc = torch.clamp(wc, 0, imgWdt-1)  # (BHW,)

        # Find batch indexes
        rep = torch.ones([height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
        bHW = torch.matmul((torch.arange(0, nbatch).float()*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

        # Box updated indexes
        W = imgWdt
        # x: W, y: H, z: D
        idx_00 = bHW + hf*W + wf
        idx_10 = bHW + hf*W + wc
        idx_01 = bHW + hc*W + wf
        idx_11 = bHW + hc*W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_00 = torch.index_select(img_flat, 0, idx_00.long())
        val_10 = torch.index_select(img_flat, 0, idx_10.long())
        val_01 = torch.index_select(img_flat, 0, idx_01.long())
        val_11 = torch.index_select(img_flat, 0, idx_11.long())

        dHeight = hc.float() - H_upmesh
        dWidth  = wc.float() - W_upmesh

        wgt_00 = (dHeight*dWidth).unsqueeze_(1)
        wgt_10 = (dHeight * (1-dWidth)).unsqueeze_(1)
        wgt_01 = ((1-dHeight) * dWidth).unsqueeze_(1)
        wgt_11 = ((1-dWidth) * (1-dHeight)).unsqueeze_(1)

        output = val_00*wgt_00 + val_10*wgt_10 + val_01*wgt_01 + val_11*wgt_11
        output = output.view(nbatch, height, width, nch).permute(0, 3, 1, 2)  #B, C, H, W
        return output


class Cblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Cblock, self).__init__()
        self.block = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

    def forward(self, x):
        return self.block(x)

class CRblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(CRblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.block(x)

class inblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super(inblock, self).__init__()
        self.block = CRblock(in_ch, out_ch, stride=stride)

    def forward(self, x):
        return self.block(x)

class outblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, output_padding=1):
        super(outblock, self).__init__()
        self.block = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)

    def forward(self, x):
        x = self.block(x)
        return x

class downblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downblock, self).__init__()
        self.block = CRblock(in_ch, out_ch, stride=2)

    def forward(self, x):
        return self.block(x)

class upblock(nn.Module):
    def __init__(self, in_ch, CR_ch, out_ch):
        super(upblock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, in_ch, 3, padding=1, stride=2, output_padding=1)
        self.block = CRblock(CR_ch, out_ch)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat([x2, upconved], dim=1)
        return self.block(x)

class registUnetBlock(nn.Module):
    def __init__(self, input_nc, encoder_nc, decoder_nc):
        super(registUnetBlock, self).__init__()
        self.inconv = inblock(input_nc, encoder_nc[0], stride=1)
        self.downconv1 = downblock(encoder_nc[0], encoder_nc[1])
        self.downconv2 = downblock(encoder_nc[1], encoder_nc[2])
        self.downconv3 = downblock(encoder_nc[2], encoder_nc[3])
        self.downconv4 = downblock(encoder_nc[3], encoder_nc[4])
        self.upconv1 = upblock(encoder_nc[4], encoder_nc[4]+encoder_nc[3], decoder_nc[0])
        self.upconv2 = upblock(decoder_nc[0], decoder_nc[0]+encoder_nc[2], decoder_nc[1])
        self.upconv3 = upblock(decoder_nc[1], decoder_nc[1]+encoder_nc[1], decoder_nc[2])
        self.keepblock = CRblock(decoder_nc[2], decoder_nc[3])
        self.upconv4 = upblock(decoder_nc[3], decoder_nc[3]+encoder_nc[0], decoder_nc[4])
        self.outconv = outblock(decoder_nc[4], decoder_nc[5], stride=1)
        self.spatialtransform = Dense2DSpatialTransformer()

    def forward(self, input):
        x1 = self.inconv(input)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)
        x = self.upconv1(x5, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.keepblock(x)
        x = self.upconv4(x, x1)
        flow = self.outconv(x)
        mov = (input[:, :1] + 1) / 2.0
        out = self.spatialtransform(mov, flow)
        return out, flow
