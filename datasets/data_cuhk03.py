from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from datasets.preprocessimage import PreProcessIm


class Cuhk03Dataset(Dataset):

	def __init__(self, path=None,transform=None):

		self.path = path
		self.transform = transform

		self.sample_paths = []
			
		for file in os.listdir(path):
			if file.endswith(".jpg"):
				self.sample_paths.append(file)
				
		self.sample_paths.sort()

		self.preprocessor = PreProcessIm(resize_h_w=(256, 128),im_mean=[0.486, 0.459, 0.408],im_std=[0.229, 0.224, 0.225])

	def __getitem__(self,index):
		
		sample = self.pil_loader(os.path.join(self.path,self.sample_paths[index]))
		
		# The path looks like this -1_c2s1_015101_00.jpg

		if (self.sample_paths[index][0:2] != '-1'):
			pid = int(self.sample_paths[index][0:4])
			camera = int(self.sample_paths[index][6:7])
		else:
			#junk image 
			pid = int(self.sample_paths[index][0:2])
			camera = int(self.sample_paths[index][4:5])

		if self.transform is not None:
			sample = self.transform(sample)

		return {
			'image':sample, 
			'pid':pid, 
			'camera':camera, 
			'path':self.sample_paths[index]
			}

	def __len__(self):
		return len(self.sample_paths)


	def pil_loader(self,path):
		with open(path, 'rb') as f:
			img = Image.open(f)
			img.convert('RGB')
			return self.preprocessor(np.array(img))