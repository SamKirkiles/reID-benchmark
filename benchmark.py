
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

torch.set_default_tensor_type('torch.cuda.FloatTensor')

import numpy as np
from model import Model 
from model_wrapper import ModelWrapper
from huanghoujing import HuangHoujingPytorch
from reid_model import ReIDModel
from scipy.spatial import distance
from datasets.data_market1501 import Market1501Dataset

def main():

	print("Starting Evaluation")

	dataset_dir ='/home/skirki/Desktop/Market-1501/raw/Market-1501-v15.09.15'

	# Create features for query 
	query_features = np.zeros((0,2048))
	query_pid = []
	query_cam = []

	test_features = np.zeros((0,2048))
	test_pid = []
	test_cam = []

	#ReIDModel change version for different model eval
	model = ReIDModel(version='huanghoujing')

	# Add preprocessing stops here
	preprocessing = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()])


	query_dataset = Market1501Dataset(
		dataset_dir+"/query",
		preprocessing)

	test_dataset = Market1501Dataset(
		dataset_dir+"/bounding_box_test",
		preprocessing)


	query_loader = DataLoader(query_dataset,batch_size=16,shuffle=False)
	test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)

	print("Generating features for query...")
	for i,samples in enumerate(query_loader):

		images, pids, cameras, sequences, frames, paths = samples['image'], samples['pid'], samples['camera'], samples['sequence'], samples['frame'], samples['path']

		forward = model.forward(images.cuda())[0]
		batch_size = images.shape[0]

		np_batch = forward.data.cpu().numpy()
		query_features = np.concatenate((query_features,np_batch),0)
		query_pid.extend(pids.data.cpu().numpy().tolist())
		query_cam.extend(cameras.data.cpu().numpy().tolist())

	print(query_features.shape)

	print("Done.")

	print("Generating features for gallery...")

	for i,features in enumerate(test_loader):
		images, pids, cameras, sequences, frames, paths = samples['image'], samples['pid'], samples['camera'], samples['sequence'], samples['frame'], samples['path']

		forward = model.forward(images.cuda())[0]
		batch_size = images.shape[0]

		np_batch = forward.data.cpu().numpy()
		test_features = np.concatenate((test_features,np_batch),0)
		test_pid.extend(pids.data.cpu().numpy().tolist())
		test_cam.extend(cameras.data.cpu().numpy().tolist())

	print(test_features.shape)
	print("Done.")
	test_features[0] = query_features[0]
	#Euclidean distance between each query feature vector and each gallery feature vector
	query_distances = distance.cdist(query_features,test_features,'euclidean')
	print(query_distances[0])
	
	avg_precision = np.zeros((query_features.shape[0],1))

	argsorted = np.argsort(query_distances,axis=1)[0]
	print(argsorted)
	print(np.array(query_distances[0])[argsorted])
	print(query_pid)
	# Calculate average precision for each query image

	print("mAP:"+ str(np.mean(avg_precision)) +" %")

if __name__ == "__main__": main()

