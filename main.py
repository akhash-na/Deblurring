import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from dataset import Dataset
from model import Generator, Adversary
from train import Trainer


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-seed', type=int, default=512, help='random seed')
	parser.add_argument('-data_folder', type=str, default='GOPRO_Large', help='dataset location')
	parser.add_argument('-pretrained', type=str, default='', help='pretrained model location')
	parser.add_argument('-n_scales', type=int, default=3, help='multi-scale deblurring level')
	parser.add_argument('-n_resblocks', type=int, default=19, help='number of residual blocks per scale')
	parser.add_argument('-n_features', type=int, default=64, help='number of feature maps')
	parser.add_argument('-kernel_size', type=int, default=5, help='size of conv kernel')
	parser.add_argument('-patch_size', type=int, default=256, help='training patch size')
	parser.add_argument('-batch_size', type=int, default=8, help='batch size for training')
	parser.add_argument('-validate_every', type=int, default=50, help='validation at every N epochs')
	parser.add_argument('-do_train', type=bool, default=True, help='train the model')
	parser.add_argument('-do_validate', type=bool, default=False, help='validate the model')
	parser.add_argument('-do_test', type=bool, default=True, help='test the model')
	parser.add_argument('-adv_loss_weight', type=float, default=1e-4, help='lambda of adversarial loss')
	parser.add_argument('-save_dir', type=str, default='models', help='directory to save models')
	parser.add_argument('-save_every', type=int, default=50, help='save state at every N epochs')
	parser.add_argument('-n_epochs', type=int, default=1000, help='number of epochs to train')
	parser.add_argument('-alternating', type=bool, default=False, help='train alternating adversary and generator')
	parser.add_argument('-gen_warmup_epochs', type=int, default=250, help='generator warmup for alternating training')
	parser.add_argument('-adv_warmup_epochs', type=int, default=250, help='adversary warmup for alternating training')
	parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('-milestones', type=int, nargs='+', default=[500, 750, 900], help='milestones for learning rate decay')
	parser.add_argument('-gamma', type=float, default=0.5, help='learning rate decay factor')
	args = parser.parse_args()

	if args.seed < 0:
		args.seed = int(time.time())

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	if args.do_train:
		train_dataset = Dataset(args, 'train')
		train_loader = DataLoader(
						dataset=train_dataset,
						batch_size=args.batch_size,
						shuffle=False,
						sampler=RandomSampler(train_dataset, replacement=True),
						num_workers=8,
						pin_memory=True,
						drop_last=True,
					)
	else:
		train_loader = None

	if args.do_validate:
		val_dataset = Dataset(args, 'val')
		val_loader = DataLoader(
						dataset=val_dataset,
						batch_size=1,
						shuffle=False,
						sampler=SequentialSampler(val_dataset),
						num_workers=8,
						pin_memory=True,
						drop_last=False,
					)
	else:
		val_loader = None

	if args.do_test:
		test_dataset = Dataset(args, 'test')
		test_loader = DataLoader(
						dataset=test_dataset,
						batch_size=1,
						shuffle=False,
						sampler=SequentialSampler(test_dataset),
						num_workers=8,
						pin_memory=True,
						drop_last=False,
					)
	else:
		test_loader = None

	dataset = {'train': train_loader, 'val': val_loader, 'test': test_loader}
	gen = Generator(args.n_resblocks, args.n_features, args.kernel_size, args.n_scales).cuda()
	adv = Adversary(args.n_features, args.kernel_size).cuda()
	optim_adv = optim.Adam(adv.parameters(), lr=args.lr)
	scheduler_adv = lrs.MultiStepLR(optim_adv, milestones=args.milestones, gamma=args.gamma)
	optim_gen = optim.Adam(gen.parameters(), lr=args.lr)
	scheduler_gen = lrs.MultiStepLR(optim_gen, milestones=args.milestones, gamma=args.gamma)
	optimizer = {'adv':optim_adv, 'gen':optim_gen}
	scheduler = {'adv':scheduler_adv, 'gen':scheduler_gen}
	model = {'adv':adv, 'gen':gen}
	trainer = Trainer(args, model, optimizer, scheduler, dataset)
	
	if args.do_train:
		trainer.train()
		trainer.plot()

	if args.do_test:
		trainer.evaluate('test')

	trainer.save()
