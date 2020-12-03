import torch
import torch.nn as nn
import os

class ResBlock(nn.Module):
	def __init__(self, n_feats, kernel_size):
		super(ResBlock, self).__init__()

		self.resblock = nn.Sequential()
		self.resblock.add_module('conv_1', nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)))
		self.resblock.add_module('relu', nn.ReLU())
		self.resblock.add_module('conv_2', nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)))

	def forward(self, x):
		y = self.resblock(x)
		y += x
		return y

class ScaleBody(nn.Module):
	def __init__(self, in_channels, out_channels, n_feats, kernel_size, n_resblocks):
		super(ScaleBody, self).__init__()

		self.body = nn.Sequential()
		self.body.add_module('first_conv', nn.Conv2d(in_channels, n_feats, kernel_size, padding=(kernel_size // 2)))
		for i in range(n_resblocks):
			self.body.add_module('resblock_' + str(i), ResBlock(n_feats, kernel_size))
		self.body.add_module('last_conv', nn.Conv2d(n_feats, out_channels, kernel_size, padding=(kernel_size // 2)))

	def forward(self, x):
		y = self.body(x)
		return y

class Generator(nn.Module):
	def __init__(self, n_resblocks, n_feats, kernel_size, n_scales):
		super(Generator, self).__init__()

		self.n_scales = n_scales

		self.scales = nn.ModuleList([ScaleBody(3, 3, n_feats, kernel_size, n_resblocks)])
		for _ in range(1, self.n_scales):
			self.scales.append(ScaleBody(6, 3, n_feats, kernel_size, n_resblocks))

		self.upscalers = nn.ModuleList([])
		for _ in range(1, self.n_scales):
			self.upscalers.append(nn.Sequential(nn.Conv2d(3, 12, kernel_size, padding=(kernel_size // 2)), nn.PixelShuffle(2)))

	def forward(self, input_pyramid):
		for i in range(len(input_pyramid)):
			input_pyramid[i] -= 128

		output_pyramid = [None] * self.n_scales
		x = input_pyramid[0]

		for i in range(self.n_scales):
			output_pyramid[i] = self.scales[i](x)
			if i+1 < len(input_pyramid):
				upscaled = self.upscalers[i](output_pyramid[i]).clone().detach()
				upscaled.requires_grad = True
				x = torch.cat((input_pyramid[i+1], upscaled), 1)

		for i in range(len(output_pyramid)):
			output_pyramid[i] += 128

		return output_pyramid

class Adversary(nn.Module):
	def __init__(self, n_feats, kernel_size):
		super(Adversary, self).__init__()

		self.adv = nn.Sequential(
			nn.Conv2d(3, n_feats//2, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats//2, n_feats//2, kernel_size, stride=2, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats//2, n_feats, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats, n_feats, kernel_size, stride=2, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats, n_feats*2, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*2, n_feats*2, kernel_size, stride=4, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*2, n_feats*4, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*4, n_feats*4, kernel_size, stride=4, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*4, n_feats*8, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*8, n_feats*8, 4, stride=4, padding=0, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*8, 1, 1, bias=False)
		)

	def forward(self, x):
		y = self.adv(x).clone().detach()
		y.requires_grad = True
		return y

class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()
		self.gen = Generator(args.n_resblocks, args.n_features, args.kernel_size, args.n_scales)
		self.adv = Adversary(args.n_features, args.kernel_size)
		self.BCELoss = nn.BCEWithLogitsLoss()
		self.MSELoss = nn.MSELoss()
		self.lamda = args.adv_loss_weight

	def forward(self, blur, sharp=None):
		fake = self.gen(blur)
		if sharp is not None:
			fake_pred = self.adv(fake[-1])
			real_pred = self.adv(sharp[-1])

			label_fake = torch.zeros_like(fake_pred)
			label_real = torch.ones_like(real_pred)

			adv_loss = self.BCELoss(fake_pred, label_fake) + self.BCELoss(real_pred, label_real)
			gen_adv_loss = self.BCELoss(fake_pred, label_real)
			gen_mse_loss = 0
			for i in range(len(fake)):
				gen_mse_loss += self.MSELoss(fake[i], sharp[i])
			gen_loss = gen_adv_loss * self.lamda + gen_mse_loss
		else:
			gen_loss = None
			adv_loss = None

		return fake, gen_loss, adv_loss