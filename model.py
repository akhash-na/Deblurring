import torch
import torch.nn as nn
import os

class ResBlock(nn.Module):
	def __init__(self, n_feats, kernel_size):
		super(ResBlock, self).__init__()

		self.resblock = nn.Sequential()
		self.resblock.add_module(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)))
		self.resblock.add_module(nn.ReLU())
		self.resblock.add_module(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)))

	def forward(self, x):
		y = self.resblock(x)
		y += x
		return y

class ScaleBody(nn.Module):
	def __init__(self, in_channels, out_channels, n_feats, kernel_size, n_resblocks):
		super(ScaleBody, self).__init__()

		self.body = nn.Sequential()
		self.body.add_module(nn.Conv2d(in_channels, n_feats, kernel_size, padding=(kernel_size // 2)))
		for _ in range(self.n_resblocks):
			self.body.add_module(ResBlock(n_feats, kernel_size))
		self.body.add_module(nn.Conv2d(n_feats, out_channels, kernel_size, padding=(kernel_size // 2)))

	def forward(self, x):
		y = self.body(x)
		return y

class Generator(nn.Module):
	def __init__(self, n_resblocks, n_feats, kernel_size, n_scales):
		super(Generator, self).__init__()

		self.n_scales = n_scales

		self.scales = [ScaleBody(3, 3, n_feats, kernel_size, n_resblocks)]
		for _ in range(1, self.n_scales):
			self.scales.append(ScaleBody(6, 3, n_feats, kernel_size, n_resblocks))

		self.upscalers = []
		for _ in range(1, self.n_scales):
			self.upscalers.append(nn.Sequential([nn.Conv2d(3, 12, kernel_size, padding=(kernel_size // 2)), nn.PixelShuffle(2)]))

	def forward(self, input_pyramid):
		for i in range(len(input_pyramid)):
			input_pyramid[i] -= 128

		output_pyramid = []
		x = input_pyramid[0]
		for i in range(self.n_scales-1):
			output_pyramid.append(self.scales[i](x))
			if i+1 < len(input_pyramid):
				upscaled = self.upscalers[i](output_pyramid[i])
				x = torch.cat((input_pyramid[i+1], upscaled), 1)

		for i in range(len(output_pyramid)):
			output_pyramid[i] += 128

		return output_pyramid

class Adversary(nn.Module):
	def __init__(self, n_feats, kernel_size):
		super(Adversary, self).__init__()

		self.adv = nn.Sequential([
			nn.Conv2d(3, n_feats, n_feats//2, kernel_size, stride=n_feats//2, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats//2, n_feats//2, kernel_size, stride=n_feats//2, padding=2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats//2, n_feats, kernel_size, stride=n_feats, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats, n_feats, kernel_size, stride=n_feats, padding=2, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats, n_feats*2, kernel_size, stride=n_feats*2, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*2, n_feats*2, kernel_size, stride=n_feats*2, padding=4, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*2, n_feats*4, kernel_size, stride=n_feats*4, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*4, n_feats*4, kernel_size, stride=n_feats*4, padding=4, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*4, n_feats*8, kernel_size, stride=n_feats*8, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(n_feats*8, n_feats*8, kernel_size, stride=n_feats*8, padding=4, bias=False),
			nn.LeakyReLU(negative_slope=0.2),
			self.dense = nn.Conv2d(n_feats*8, 1, 1, bias=False)
		])

	def forward(self, x):
		y = adv(x)
		return y

class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()

		self.args = args
		self.save_dir = os.path.join(args.save_dir, 'models')
		os.makedirs(self.save_dir, exist_ok=True)

		self.gen = Generator(args.n_resblocks, args.n_feats, args.kernel_size, args.n_scales)
		if self.args.loss.lower().find('adv') >= 0:
			self.adv = Adversary(args.n_feats, args.kernel_size)
		else:
			self.adv = None

		self.load(path=args.pretrained)

	def forward(self, input):
		return self.model.G(input)

	def save(self, epoch):
		torch.save(self.state_dict(), os.path.join(self.save_dir, 'model-{:d}.pt'.format(epoch)))

	def load(self, epoch=None, path=None):
		if epoch:
			model = os.path.join(self.save_dir, 'model-{:d}.pt'.format(epoch))
		elif path:
			model = path
		else:
			raise Exception('no epoch number or model path specified')

		print('Loading model from {}'.format(model))
		state_dict = torch.load(model)
		self.load_state_dict(state_dict)