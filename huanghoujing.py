import torch
from model_wrapper import ModelWrapper
from model import Model 
import numpy as np

class HuangHoujingPytorch(ModelWrapper):

	def __init__(self):

		self.model = Model.Model(128,751) 
		self.model.load_state_dict(torch.load('weights/model_weight.pth'))
		self.model = self.model.eval()

	def forward(self,x):
		return self.model(x)
