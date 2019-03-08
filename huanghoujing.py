import torch
from model_wrapper import ModelWrapper
from models.AlignedReID_huanghoujing.aligned_reid.dataset.PreProcessImage import PreProcessIm
from models.AlignedReID_huanghoujing.aligned_reid.model import Model
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class HuangHoujingPytorch(ModelWrapper):

	def __init__(self):

		self.model = Model.Model(128,751)
		self.model.load_state_dict(torch.load('weights/model_weight.pth'))
		self.model = self.model.eval()
		self.model.cuda()
		
		self.transform = None
		self.preprocessor = PreProcessIm(resize_h_w=(256, 128),im_mean=[0.486, 0.459, 0.408],im_std=[0.229, 0.224, 0.225])
		
	def forward(self,x):
		return self.model(x)[0]
