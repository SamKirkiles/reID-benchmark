import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from model import Model 
from model_wrapper import ModelWrapper
from huanghoujing import HuangHoujingPytorch
from reid_model import ReIDModel

def main():

	train_dir ='/home/skirki/Desktop/Market-1501/'

	# Preprocessing
	train_dataset = datasets.ImageFolder(
		train_dir,
		transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()
		]))

	#ReIDModel Class
	model = ReIDModel(version='huanghoujing')

	eval_loader = DataLoader(train_dataset, batch_size=4,shuffle=True)

	for x,y in enumerate(eval_loader):
		print(x)

if __name__ == "__main__": main()

