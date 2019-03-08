import torch
import os
from model_wrapper import ModelWrapper
from models.DARENet.models import dare_models
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import torch.nn.parallel

torch.set_default_tensor_type('torch.FloatTensor')

class DareNetPytorch(ModelWrapper):

	def __init__(self):
		
		self.preprocessor = None
		
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		
		crop_size = [256, 128]
		gap_size = [x // 32 for x in crop_size]
		scale_image_size = [int(x * 1.125) for x in crop_size]

		self.transform = transforms.Compose([
        transforms.Resize(scale_image_size),
        transforms.ToTensor(),
        normalize, ])
		
		model  = getattr(dare_models, 'dare_R')(pretrained=True, gap_size=gap_size, gen_stage_features=False)
		model = nn.DataParallel(model).cuda()
		
		checkpoint = torch.load('weights/market1501_res50.pth.tar')
		model.load_state_dict(checkpoint['state_dict'])
		self.model = model.eval()

	def forward(self,x):
		
		return self.model(x)