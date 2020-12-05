import os
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

class Trainer():

	def __init__(self, args, model, optimizer, scheduler, dataset):
		self.args = args
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.dataset = dataset
		self.plotloss = []
		if args.pretrained != '':
			checkpoint = torch.load(args.pretrained)
			self.model['adv'].load_state_dict(checkpoint['adv_model'])
			self.model['gen'].load_state_dict(checkpoint['gen_model'])
			self.optimizer['adv'].load_state_dict(checkpoint['adv_optimizer'])
			self.optimizer['gen'].load_state_dict(checkpoint['gen_optimizer'])
			self.scheduler['adv'].load_state_dict(checkpoint['adv_lrs'])
			self.scheduler['gen'].load_state_dict(checkpoint['gen_lrs'])
			self.plotloss = checkpoint['plotloss']

	def save(self, epoch=None):
		epoch = self.args.n_epochs if epoch is None else epoch  
		checkpoint = { 
			'epoch': epoch,
			'adv_model': self.model['adv'].state_dict(),
			'gen_model': self.model['gen'].state_dict(),
			'adv_optimizer': self.optimizer['adv'].state_dict(),
			'gen_optimizer': self.optimizer['gen'].state_dict(),
			'adv_lrs': self.scheduler['adv'].state_dict(),
			'gen_lrs': self.scheduler['gen'].state_dict(),
			'plotloss': self.plotloss,
			}
		torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-epoch-'+str(epoch)+'.pt'))

	def train(self):
		self.model['adv'].train()
		self.model['gen'].train()

		BCE = nn.BCEWithLogitsLoss()
		MSE = nn.MSELoss()

		for epoch in range(self.args.n_epochs):
			gen_loss_t = 0.0
			adv_loss_t = 0.0
			ct = 0

			torch.set_grad_enabled(True)

			print('[Epoch %d / lr %f]' % (epoch + 1, self.scheduler['gen'].get_last_lr()[-1]))
			tq = tqdm(self.dataset['train'], ncols=80, smoothing=0, bar_format='{desc}|{bar}{r_bar}')

			for idx, batch in enumerate(tq):
				blur = batch[0]
				sharp = batch[1]

				for i in range(len(blur)):
					blur[i] = blur[i].cuda()
					sharp[i] = sharp[i].cuda()

				fake = self.model['gen'](blur)

				fake_adv = fake[-1].detach()

				fake_pred = self.model['adv'](fake_adv)
				real_pred = self.model['adv'](sharp[-1])

				fake_label = torch.zeros_like(fake_pred)
				real_label = torch.ones_like(real_pred)

				adv_loss = BCE(fake_pred, fake_label) + BCE(real_pred, real_label)

				if not self.args.alternating or (self.args.alternating and epoch >= self.args.gen_warmup_epochs): 
					self.optimizer['adv'].zero_grad()
					adv_loss.backward()
					self.optimizer['adv'].step()

				gen_loss = self.args.adv_loss_weight * BCE(self.model['adv'](fake[-1]), real_label)
				for i in range(len(fake)):
					gen_loss += MSE(fake[i], sharp[i])

				if not self.args.alternating or (self.args.alternating and (epoch < self.args.gen_warmup_epochs or epoch >= self.args.gen_warmup_epochs + self.args.adv_warmup_epochs)):
					self.optimizer['gen'].zero_grad()
					gen_loss.backward()
					self.optimizer['gen'].step()

				gen_loss_t += gen_loss.item()
				adv_loss_t += adv_loss.item()
				ct += 1

				tq.set_description('[Adv Loss: %.3f / Gen Loss: %.3f]' % (adv_loss_t / ct, gen_loss_t / ct))

			self.plotloss.append(adv_loss_t / ct)

			self.scheduler['adv'].step()
			self.scheduler['gen'].step()

			if (epoch + 1) % self.args.save_every == 0:
				self.save(epoch + 1)

			if (epoch + 1) % self.args.validate_every == 0 and self.args.do_validate:
				self.evaluate('val')

	def evaluate(self, mode):
		self.model['adv'].eval()
		self.model['gen'].eval()

		BCE = nn.BCEWithLogitsLoss()
		MSE = nn.MSELoss()

		gen_loss_t = 0.0
		psnr_t = 0.0
		ssim_t = 0.0
		ct = 0

		torch.set_grad_enabled(False)       
		tq = tqdm(self.dataset[mode], ncols=80, smoothing=0, bar_format='{desc}|{bar}{r_bar}')

		for idx, batch in enumerate(tq):
			blur = batch[0]
			sharp = batch[1]

			for i in range(len(blur)):
				blur[i] = blur[i].cuda()
				sharp[i] = sharp[i].cuda()
		
			fake = self.model['gen'](blur)
			real_pred = self.model['adv'](sharp[-1])
			real_label = torch.ones_like(real_pred)

			gen_loss = self.args.adv_loss_weight * BCE(self.model['adv'](fake[-1]), real_label)
			for i in range(len(fake)):
				gen_loss += MSE(fake[i], sharp[i])
			
			gen_loss_t += gen_loss.item()

			sharpl = sharp[-1].clamp(0, 255).round_().cpu().detach().numpy()
			fakel = fake[-1].clamp(0, 255).round_().cpu().detach().numpy()
			batch_size = len(fakel)

			psnr_l = 0
			ssim_l = 0
			for i in range(batch_size):
				im1 = np.moveaxis(sharpl[i], 0, -1)
				im2 = np.moveaxis(fakel[i], 0, -1)
				psnr_l += psnr(im1, im2, data_range=255)
				ssim_l += ssim(im1, im2, data_range=255, multichannel=True)
			psnr_l /= batch_size
			ssim_l /= batch_size

			psnr_t += psnr_l
			ssim_t += ssim_l
			ct += 1

			tq.set_description('[Loss: %.3f / PSNR: %.3f / SSIM: %.3f]' 
				% (gen_loss_t / ct, psnr_t / ct, ssim_t / ct))

	def plot(self):
		plt.plot(range(1, len(self.plotloss)+1), self.plotloss)
		plt.savefig(os.path.join(self.args.save_dir, 'loss.png'))
