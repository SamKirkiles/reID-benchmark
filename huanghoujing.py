import torch
from model_wrapper import ModelWrapper
from model import Model 
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class HuangHoujingPytorch(ModelWrapper):

	def __init__(self):

		self.model = Model.Model(128,751) 
		self.model.load_state_dict(torch.load('weights/model_weight.pth'))
		self.model = self.model.eval()
		self.model.cuda()
	def forward(self,x):
		return self.model(x)
