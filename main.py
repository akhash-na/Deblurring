import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from .dataset import Dataset
from .model import Model
from .train import Trainer


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-seed', type=int, default=512, help='random seed')
	parser.add_argument('-data_folder', type=str, default='/GOPRO_Large', help='dataset location')
	parser.add_argument('-pretrained', type=str, default='', help='pretrained model location')
	parser.add_argument('-n_scales', type=int, default=3, help='multi-scale deblurring level')
	parser.add_argument('-n_resblocks', type=int, default=19, help='number of residual blocks per scale')
	parser.add_argument('-n_features', type=int, default=64, help='number of feature maps')
	parser.add_argument('-kernel_size', type=int, default=5, help='size of conv kernel')
	parser.add_argument('-patch_size', type=int, default=256, help='training patch size')
	parser.add_argument('-batch_size', type=int, default=2, help='batch size for training')
	parser.add_argument('-validate_every', type=int, default=10, help='validation at every N epochs')
	parser.add_argument('-do_train', type=str2bool, default=True, help='train the model')
	parser.add_argument('-do_validate', type=str2bool, default=True, help='validate the model')
	parser.add_argument('-do_test', type=str2bool, default=True, help='test the model')
	parser.add_argument('-adv_loss_weight', type=float, default=1e-4, help='lambda of adversarial loss')
	parser.add_argument('-save_dir', type=str, default='', help='directory to save models')
	parser.add_argument('-save_every', type=int, default=10, help='save state at every N epochs')
	parser.add_argument('-n_epochs', type=int, default=1000, help='number of epochs to train')
	parser.add_argument('-train_adv_only', type=str2bool, default=False, help='to train only the adversary')
	args = parser.parse_args()

	os.makedir(args.save_dir, exist_ok=True)

	if args.seed < 0:
		args.seed = int(time.time())

	if args.do_train:
		train_dataset = Dataset(args, 'train')
		train_loader = DataLoader(
						dataset=train_dataset,
						batch_size=args.batch_size,
						shuffle=False,
						sampler=RandomSampler(train_dataset, replacement=True),
						pin_memory=True,
						drop_last=True,
					)
	else:
		train_loader = None

	if args.do_validate:
		val_dataset = Dataset(args, 'val')
		val_loader = DataLoader(
						dataset=val_dataset,
						batch_size=args.batch_size,
						shuffle=False,
						sampler=SequentialSampler(val_dataset),
						pin_memory=True,
						drop_last=False,
					)
	else:
		val_loader = None

	if args.do_test:
		test_dataset = Dataset(args, 'test')
		test_loader = DataLoader(
						dataset=test_dataset,
						batch_size=args.batch_size,
						shuffle=False,
						sampler=SequentialSampler(test_dataset),
						pin_memory=True,
						drop_last=False,
					)
	else:
		test_loader = None

	dataset = {'train': train_loader, 'val': val_loader, 'test': test_loader}
	model = Model(args)
	optim_adv = optim.Adam(model.adv.parameters(), lr=1e-4)
	scheduler_adv = optim.MultiStepLR(optim_adv, milestones=[500, 750, 900], gamma=0.1)
	optim_gen = optim.Adam(model.gen.parameters(), lr=1e-4)
	scheduler_gen = optim.MultiStepLR(optim_gen, milestones=[500, 750, 900], gamma=0.1)
	optimizer = {'adv':optim_adv, 'gen':optim_gen}
	scheduler = {'adv':scheduler_adv, 'gen':scheduler_gen}
	trainer = Trainer(args, model, optimizer, scheduler, dataset)
	
	trainer.train()

	if args.do_test:
		trainer.evaluate('test')

	trainer.save()
