import os
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np

class Trainer():

	def __init__(self, args, model, optimizer, scheduler, dataset):
		self.args = args
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.dataset = dataset
		if args.pretrained != '':
			checkpoint = torch.load(args.pretrained)
			self.model.load_state_dict(checkpoint['model'])
			self.optimizer['adv'].load_state_dict(checkpoint['adv_optimizer'])
			self.optimizer['gen'].load_state_dict(checkpoint['gen_optimizer'])
			self.scheduler['adv'].load_state_dict(checkpoint['adv_lrs'])
			self.scheduler['gen'].load_state_dict(checkpoint['gen_lrs'])

	def save(self, epoch=None):
		epoch = self.args.n_epochs if epoch is None else epoch  
		checkpoint = { 
			'epoch': epoch,
			'model': self.model.state_dict(),
			'adv_optimizer': self.optimizer['adv'].state_dict(),
			'gen_optimizer': self.optimizer['gen'].state_dict(),
			'adv_lrs': self.scheduler['adv'].state_dict(),
			'gen_lrs': self.scheduler['gen'].state_dict(),
			}
		torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-epoch-'+str(epoch)+'.pt'))

	def train(self):
		self.model.train()

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

				fake, gen_loss, adv_loss = self.model(blur, sharp)

				if not self.args.alternating or (self.args.alternating and epoch >= self.args.gen_warmup_epochs): 
					self.optimizer['adv'].zero_grad()
					adv_loss.backward()
					self.optimizer['adv'].step()
				
				if not self.args.alternating or (self.args.alternating and (epoch < self.args.gen_warmup_epochs or epoch >= self.args.gen_warmup_epochs + self.args.adv_warmup_epochs)):
					self.optimizer['gen'].zero_grad()
					gen_loss.backward()
					self.optimizer['gen'].step()

				gen_loss_t += gen_loss.item()
				adv_loss_t += adv_loss.item()
				ct += 1

				tq.set_description('[Adv Loss: %.3f / Gen Loss: %.3f]' % (adv_loss_t / ct, gen_loss_t / ct))

			self.scheduler['adv'].step()
			self.scheduler['gen'].step()

			print('[Epoch %d / Adv Loss: %.3f / Gen Loss: %.3f]' % (epoch + 1, adv_loss_t / ct, gen_loss_t / ct))

			if (epoch + 1) % self.args.save_every == 0:
				self.save(epoch + 1)

			if (epoch + 1) % self.args.validate_every == 0 and self.args.do_validate:
				self.evaluate('val')

	def evaluate(self, mode):
		self.model.eval()

		gen_loss_t = 0.0
		adv_loss_t = 0.0
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
			
			fake, gen_loss, adv_loss = self.model(blur, sharp)
			gen_loss_t += gen_loss.item()
			adv_loss_t += adv_loss.item()
			ct += 1

			im1 = np.moveaxis(sharp[-1].clamp(0, 255).round_().cpu().detach().numpy()[0], 0, -1)
			im2 = np.moveaxis(fake[-1].clamp(0, 255).round_().cpu().detach().numpy()[0], 0, -1)

			psnr_t += psnr(im1, im2, data_range=255)
			ssim_t += ssim(im1, im2, data_range=255, multichannel=True)

		print('[Val Adv Loss: %.3f / Val Gen Loss: %.3f / PSNR: %.3f / SSIM: %.3f]' 
			% (adv_loss_t / ct, gen_loss_t / ct, psnr_t / ct, ssim_t / ct))
