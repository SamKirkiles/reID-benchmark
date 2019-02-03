import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import Model 
from model_wrapper import ModelWrapper
from huanghoujing import HuangHoujingPytorch
from reid_model import ReIDModel

def main():

	train_dir ='/home/skirki/Desktop/Market-1501/Market-1501-v15.09.15/bounding_box_train'

	train_dataset = datasets.ImageFolder(
		train_dir,
		transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()
		]))


	model = ReIDModel(version='huanghoujing')
	model.forward()

if __name__ == "__main__": main()

