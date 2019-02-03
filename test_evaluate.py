from reid_model import ReIDModel
import argparse
import datasets
from utils.data.preprocessor import Preprocessor

from torchvision.transforms import *

import torch
from torch.utils.data import DataLoader

def evaluate(args):
	print("Starting evaluation")

	model = ReIDModel(version='huanghoujing')

	t = Compose([
		ToTensor()
		])


	dataset = datasets.create('market1501', 'data/{}'.format('market1501'))
	preprocessed = Preprocessor(list(set(dataset.query)|set(dataset.gallery)),root=dataset.images_dir,transform=t)
	dataloader = DataLoader(preprocessed, batch_size=args.batchsize, shuffle=False)

	for i, (imgs, fnames, pids, _) in enumerate(dataloader):
		print(model.forward(imgs)[0].size())

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Evaluate")
	parser.add_argument('-b', '--batchsize', type=int, default=50)
	evaluate(parser.parse_args())
