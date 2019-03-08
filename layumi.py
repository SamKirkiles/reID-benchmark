import torch
import os
from model_wrapper import ModelWrapper
from models.Person_reID_baseline_pytorch.model import ft_net
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

torch.set_default_tensor_type('torch.FloatTensor')

class LayumiPytorch(ModelWrapper):

	def __init__(self):
		
		self.preprocessor = None
		self.transform = transforms.Compose([
		transforms.Resize((256,128), interpolation=3),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


		model_structure = ft_net(751)
		load = torch.load('weights/ft_ResNet50/net_last.pth')
		model_structure.load_state_dict(load)

		model_structure.model.fc = nn.Sequential()
		model_structure.classifier = nn.Sequential()
		model_structure = model_structure.eval()
		self.model = model_structure.cuda()

	def load_network(network):
	    save_path = os.path.join('./weights',name,'net_%s.pth'%opt.which_epoch)
	    network.load_state_dict(torch.load(save_path))
	    return network

	def forward(self,x):
		return self.model(x)