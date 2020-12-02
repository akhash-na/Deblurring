import os
import random
import imageio
import numpy as np
import torch.utils.data as data
import common

class Dataset(data.Dataset):
	def __init__(self, args, mode):
		super(Dataset, self).__init__()
		self.blur_list = []
		self.sharp_list = []

		for path, dirs, files in os.walk(os.path.join(args.data_folder, mode)):
			for filename in files:
				filepath = os.path.join(path, filename)

				if filepath.endswith(".png"):
					if 'blur' in filepath and 'blur_gamma' not in filepath:
						blur_list.append(filepath)
					if 'sharp' in filepath:
						sharp_list.append(filepath)

	def __getitem__(self, idx):
		blur = imageio.imread(self.blur_list[idx], pilmode='RGB')
		if len(self.sharp_list) > 0:
			sharp = imageio.imread(self.sharp_list[idx], pilmode='RGB')
			imgs = [blur, sharp]
		else:
			imgs = [blur]
		pad_width = 0
		if self.mode == 'train':
			imgs = common.crop(*imgs, ps=self.args.patch_size)
			imgs = common.augment(*imgs, hflip=True, rot=True, shuffle=True, change_saturation=True, rgb_range=self.args.rgb_range)
			imgs[0] = common.add_noise(imgs[0], sigma_sigma=2, rgb_range=self.args.rgb_range)

		if self.args.gaussian_pyramid:
			imgs = common.generate_pyramid(*imgs, n_scales=self.args.n_scales)

		imgs = common.np2tensor(*imgs)

		blur = imgs[0]
		sharp = imgs[1] if len(imgs) > 1 else None

		return blur, sharp, pad_width