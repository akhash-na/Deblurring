import torch
import torch.nn as nn

class AdversarialLoss(nn.modules.loss._Loss):
	def __init__(self, model, optimizer):
		super(Adversarial, self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.BCELoss = nn.BCEWithLogitsLoss()

	def forward(self, fake, real, training=False, adv_only=False):
		if training:
			self.optimizer.adv.zero_grad()

			fake_pred = self.model.adv(fake)
			real_pred = self.model.adv(real)

			label_fake = torch.zeros_like(fake_pred)
			label_real = torch.ones_like(real_pred)

			adv_loss = self.BCELoss(fake_pred, label_fake) + self.BCELoss(real_pred, label_real)

			adv_loss.backward()
			optimizer.adv.step()
		else:
			real_pred = self.model.adv(real)
			label_real = torch.ones_like(read_pred)

		fake_pred = self.model.adv(fake)

		if adv_only:
			gen_loss = 0
		else:
			gen_loss = self.BCELoss(fake_pred, label_real)

		return gen_loss