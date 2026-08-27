import torch
import itertools
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter

from util import contextual as cl  # contextual loss


def mul_mask(image, mask):
    return (image + 1) * mask - 1

# MFR-Net
class MFRNetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(norm='batch', netG='mfrnet', dataset_mode='aligned')
        parser.set_defaults(norm='instance', netG='mfrnet', dataset_mode='aligned')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L1H', type=float, default=100.0, help='weight for L1H loss')
            parser.add_argument('--lambda_DDP', type=float, default=1, help='weight for DDP')
            parser.add_argument('--lambda_DP', type=float, default=1, help='weight for DP loss')
            parser.add_argument('--lambda_HFC_OT', type=float, default=10.0, help='weight for HFC contextual OT loss')
            parser.add_argument('--RMS', action='store_true', help='whether use RMSprop optimizer')

        # HFC filter
        parser.add_argument('--filter_width', type=int, default=53, help='kernel_len for Gaussian_kernel')
        parser.add_argument('--nsig', type=int, default=9, help='nsig for Gaussian_kernels')
        parser.add_argument('--sub_low_ratio', type=float, default=1.0, help='ratio for LFC')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.input_nc = opt.input_nc

        self.loss_names = ['DP', 'DP_fake', 'DP_real',
                           'DDP', 'DDP_fake_S', 'DDP_fake_T',
                           'G', 'G_L1', 'G_L1H', 'G_DP', 'G_DDP']
        if self.isTrain and opt.use_HFC_OT_loss:
            self.loss_names.append('G_HFC_OT')

        self.visual_names = ['real_SA', 'real_SAH', 'fake_SB', 'fake_SBH',
                             'real_SB', 'real_SBH', ]

        # 初始化 guide filter 和灰度图工具
        self.hfc_filter = HFCFilter(opt.filter_width, nsig=opt.nsig, sub_low_ratio=opt.sub_low_ratio, sub_mask=True,
                                    is_clamp=True).to(self.device)

        if self.isTrain:
            self.model_names = ['G', 'DP', 'DDP']
        else:  # during test time, only load G
            self.model_names = ['G']
            self.visual_names = ['real_TA', 'real_TAH', 'fake_TB', 'fake_TBH']

        # define the generator
        self.netG = networks.define_G(6, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define the discriminator
        if self.isTrain:
            # pixel discriminator
            self.netDP = networks.define_D(3, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # domain discriminator
            self.netDDP = networks.define_D(opt.ngf*8, opt.ndf, "style",
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # default: vanilla
            self.criterionL1 = torch.nn.L1Loss()

            # contextual loss
            if opt.use_HFC_OT_loss:  # VGG-19 (frozen)
                self.criterion_contextual = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4').to(self.device)
                self.criterion_contextual.eval()
                self.criterion_contextual.requires_grad_(False)

            # optimizers
            if not self.opt.RMS:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netDP.parameters(), self.netDDP.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr, alpha=0.9)
                self.optimizer_D = torch.optim.RMSprop(itertools.chain(self.netDP.parameters(), self.netDDP.parameters()),
                                                       lr=opt.lr, alpha=0.9)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        if self.isTrain:
            AtoB = self.opt.direction == 'AtoB'
            self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)  # degrade
            self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)  # gt
            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.S_mask = input['S_mask'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)

            self.real_SAH = self.hfc_filter(self.real_SA, self.S_mask)
            self.real_TAH = self.hfc_filter(self.real_TA, self.T_mask)
            self.real_SBH = self.hfc_filter(self.real_SB, self.S_mask)

            self.real_SA6 = torch.cat([self.real_SA, self.real_SAH], dim=1)
            self.real_TA6 = torch.cat([self.real_TA, self.real_TAH], dim=1)

            self.image_paths = input['TA_path']
        else:
            AtoB = self.opt.direction == 'AtoB'
            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.real_TAH = self.hfc_filter(self.real_TA, self.T_mask)
            self.real_TA6 = torch.cat([self.real_TA, self.real_TAH], dim=1)
            self.image_paths = input['TA_path']

    def forward(self):
        if self.isTrain:
            self.fake_SB, self.fake_SF = self.netG(self.real_SA6, encode_only=False)  # G(SA)
            self.fake_SB = mul_mask(self.fake_SB, self.S_mask)
            self.fake_SBH = self.hfc_filter(self.fake_SB, self.S_mask)
            self.fake_SBH = mul_mask(self.fake_SBH, self.S_mask)

            self.fake_TF = self.netG(self.real_TA6, encode_only=True)  # [batch_size, 512, 4, 4]
        else:
            self.fake_TB, _ = self.netG(self.real_TA6, encode_only=False)  # G(TA)
            self.fake_TB = mul_mask(self.fake_TB, self.T_mask)
            self.fake_TBH = self.hfc_filter(self.fake_TB, self.T_mask)
            self.fake_TBH = mul_mask(self.fake_TBH, self.T_mask)

    # domain discriminator loss
    def backward_DDP(self):
        pred_fake_S = self.netDDP(self.fake_SF.detach())  # [B,1]
        pred_fake_T = self.netDDP(self.fake_TF.detach())  # [B,1]

        self.loss_DDP_fake_S = self.criterionGAN(pred_fake_S, True)
        self.loss_DDP_fake_T = self.criterionGAN(pred_fake_T, False)

        self.loss_DDP = (self.loss_DDP_fake_S + self.loss_DDP_fake_T) * 0.5
        self.loss_DDP.backward()

    # pixel discriminator loss
    def backward_DP(self):
        pred_fake_SB = self.netDP(self.fake_SB.detach())
        pred_real_SB = self.netDP(self.real_SB.detach())

        self.loss_DP_fake = self.criterionGAN(pred_fake_SB, False)
        self.loss_DP_real = self.criterionGAN(pred_real_SB, True)

        self.loss_DP = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_DP.backward()

    def backward_G(self):
        # L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1
        self.loss_G_L1H = self.criterionL1(self.fake_SBH, self.real_SBH) * self.opt.lambda_L1H

        # GAN loss
        pred_fake_SB = self.netDP(self.fake_SB)
        self.loss_G_DP = self.opt.lambda_DP * self.criterionGAN(pred_fake_SB, True)

        pred_fake_T = self.netDDP(self.fake_TF)
        # pred_fake_S = self.netDDP(self.fake_SF)
        self.loss_G_DDP = self.criterionGAN(pred_fake_T, True) * self.opt.lambda_DDP

        self.loss_G = self.loss_G_L1 + self.loss_G_L1H + self.loss_G_DP + self.loss_G_DDP

        # OT loss
        if self.opt.use_HFC_OT_loss:  # default: true
            # [-1, 1] -> [0, 1]
            fake_SBH_01 = (self.fake_SBH + 1) / 2
            real_SBH_01 = (self.real_SBH + 1) / 2
            self.loss_G_HFC_OT = self.criterion_contextual(fake_SBH_01, real_SBH_01) * self.opt.lambda_HFC_OT

            self.loss_G += self.loss_G_HFC_OT

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad([self.netDP, self.netDDP], True)
        self.optimizer_D.zero_grad()
        self.backward_DP()
        self.backward_DDP()
        self.optimizer_D.step()

        self.set_requires_grad([self.netDP, self.netDDP], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
