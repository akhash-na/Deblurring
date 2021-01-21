from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import glob
import imageio
import common
import argparse
import numpy as np

import torch
import torch.nn as nn

from model import Generator

k = 2
n_resblocks = 10
n_features = 32
kernel_size = 5
n_scales = 2
path = 'models/checkpoint-epoch-100.pt'

gen = Generator(n_resblocks, n_features, kernel_size, n_scales)
checkpoint = torch.load(path, map_location=torch.device('cpu'))
gen.load_state_dict(checkpoint['gen_model'])
gen.eval()
torch.set_grad_enabled(False) 

app = Flask(__name__)

@app.route('/')
def interactive():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def process():
	if 'file' not in request.files:
		flash('Choose file')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('Choose file')
		return redirect(request.url)
	if file and (file.filename.endswith('.png') or file.filename.endswith('.jpg') or file.filename.endswith('.jpeg')):
		filename = secure_filename(file.filename)
		path = os.path.join('static', filename)
		file.save(path)
		blur_orig = imageio.imread(path, pilmode='RGB')
		blur = [blur_orig]

		blur[0], pad_width = common.pad(blur[0], divisor=2**(n_scales-1))
		blur = common.generate_pyramid(*blur, n_scales=n_scales)
		blur = common.np2tensor(*blur)[0]
		for i in range(len(blur)):
			blur[i] = blur[i].unsqueeze(0)
		sharp = gen(blur)[-1]
		sharp, _ = common.pad(sharp, pad_width=pad_width, negative=True)
		sharp_np = sharp[0].clamp(0, 255).round_().cpu().detach().numpy()
		sharp_np = np.moveaxis(sharp_np, 0, -1)

		imageio.imwrite('static/sharp_'+str(k)+'_'+filename, sharp_np)
		imageio.imwrite('static/blur_'+str(k)+'_'+filename, blur_orig)
		os.remove(path)
		return render_template('index.html', sharp='sharp_'+str(k)+'_'+filename, blur='blur_'+str(k)+'_'+filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename=filename), code=301)

if __name__ == "__main__":
    app.run()
