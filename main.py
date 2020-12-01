import argparse
import os
import time
from .dataset import Dataset
from .model import Model
from .optimizer import Optimizer
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
	parser.add_argument('-batch_size', type=int, default=16, help='batch size for training')
	parser.add_argument('-validate_every', type=int, default=10, help='validation at every N epochs')
	parser.add_argument('-do_train', type=str2bool, default=True, help='train the model')
	parser.add_argument('-do_validate', type=str2bool, default=True, help='validate the model')
	parser.add_argument('-do_test', type=str2bool, default=True, help='test the model')
	parser.add_argument('-loss', type=str, default='1*L1', help='loss function')
	parser.add_argument('-save_dir', type=str, default='', help='directory to save logs')
	parser.add_argument('-save_every', type=int, default=10, help='save state at every N epochs')
	parser.add_argument('-n_epochs', type=int, default=1000, help='number of epochs to train')
	args = parser.parse_args()

	os.makedir(args.save_dir, exist_ok=True)

	if args.seed < 0:
		args.seed = int(time.time())

	dataset = Dataset(args).get_loader()
	model = Model(args)
	optimizer = Optimizer(args, model)
	trainer = Trainer(args, model, loss, optimizer, dataset)

	if args.do_train:
		for epoch in range(args.n_epochs):
			trainer.train(epoch)

			if args.do_validate and epoch % args.validate_every == 0:
				trainer.validate(epoch)

			if epoch % args.save_every == 0:
				trainer.save(args.save_dir, epoch)

	if args.do_test:
		trainer.test(args.n_epochs)

	trainer.save(save_dir, args.n_epochs)
