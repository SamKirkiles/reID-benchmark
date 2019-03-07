import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import pickle

torch.set_default_tensor_type('torch.cuda.FloatTensor')

import numpy as np
 
from model_wrapper import ModelWrapper
from huanghoujing import HuangHoujingPytorch
from reid_model import ReIDModel
from scipy.spatial import distance
from datasets.data_market1501 import Market1501Dataset
from datasets.data_cuhk03 import Cuhk03Dataset
from re_ranking  import re_ranking

def main():

	query_disp_choice = 2

	dist_save_path = "distance_matrix.p"
	query_save_path = "query_features.p"
	gallery_save_path = "gallery_features.p"

	# Change this to your unzipped Market1501
	#dataset_dir ='data/Market-1501-v15.09.15'
	dataset_dir = 'data/cuhk03-np/labeled'

	print("Starting Evaluation")

	#ReIDModel change version for different model eval
	model = ReIDModel(version='huanghoujing')

	#query_dataset = Market1501Dataset(
	#	dataset_dir+"/security_query",
	#	None)

	query_dataset = Cuhk03Dataset(
		dataset_dir +"/query",
		None)
		
	#test_dataset = Market1501Dataset(
	#	dataset_dir+"/bounding_box_test",
	#	None)
	
	test_dataset = Cuhk03Dataset(
		dataset_dir+"/bounding_box_test",
		None)

	query_loader = DataLoader(query_dataset,batch_size=16,shuffle=False)
	test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)

	print("Retreiving features for query...")
	query_features, query_pid, query_cam, query_path = generate_features(model,query_loader,query_save_path)
	print("Done.")

	print("Retreiving features for gallery")
	test_features, test_pid, test_cam, test_path = generate_features(model,test_loader, gallery_save_path)
	print("Done.")

	print("Calculating distances")

	#Euclidean distance between each query feature vector and each gallery feature vector
	if dist_save_path is not None and os.path.exists(dist_save_path):
		print("Loading distance matrix from save: " + dist_save_path)
		q_g_distances,q_q_distances,g_g_distances =  pickle.load( open( dist_save_path, "rb" ) )
	else:
		print("No save detected. Calculauting new distance matrix.")
		q_g_distances = distance.cdist(query_features,test_features,'euclidean')
		q_q_distances = distance.cdist(query_features,query_features,'euclidean')
		g_g_distances = distance.cdist(test_features,test_features,'euclidean')
		if dist_save_path is not None:
			print("Saving features...")
			pickle.dump( (q_g_distances,q_q_distances,g_g_distances), open( dist_save_path, "wb" ) )
		else:
			print("Could not save features as path was not specified.")

	print("Query Person ID: " + str(query_pid[query_disp_choice]))
	print("Top 20 gallery matches: ")

	avg_precision = np.zeros(query_features.shape[0])
	r1 = []

	sorted_ind = np.argsort(q_g_distances,axis=1)
	
	print("Re-Ranking...")
	re_ranked_q_g_distances = re_ranking(q_g_distances,q_q_distances,g_g_distances)
	print("Done.")
	
	
	
	sorted_ind = np.argsort(re_ranked_q_g_distances,axis=1)
	print(re_ranked_q_g_distances[0][sorted_ind[0]])
	print(re_ranked_q_g_distances[1][sorted_ind[1]])
	print(re_ranked_q_g_distances[2][sorted_ind[2]])

	for k in range(0,re_ranked_q_g_distances.shape[0]):
		
		# junk images with pid == -1 after sorting
		junk_images_pid = np.where(np.array(test_pid)[sorted_ind[k]] == -1)[0]
		junk_images_cam = np.where(np.array(test_cam)[sorted_ind[k]] == query_cam[k])[0]
		junk_images = np.concatenate((junk_images_pid,junk_images_cam))
		good_images = np.delete(np.arange(len(test_pid)),junk_images)
				
		if np.array(test_pid)[sorted_ind[k]][good_images][0] == query_pid[k]:
			r1.append(1)
		else:
			r1.append(0)

		binary_labels = np.array(test_pid)[sorted_ind[k]][good_images] == query_pid[k]
		avg_precision[k] = calculate_ap(binary_labels)
		
		if query_disp_choice == k:
			result_paths = np.array(test_path)[sorted_ind][k][good_images][0:20]
			result_pid = np.array(test_pid)[sorted_ind][k][good_images][0:20]
			result_cam = np.array(test_cam)[sorted_ind][k][good_images][0:20]
			result_dist = re_ranked_q_g_distances[k][sorted_ind[k]][good_images][0:20]
			
			display(query_path[query_disp_choice],(result_paths,result_pid,result_cam,result_dist),dataset_dir)

	# Calculate average precision for each query image

	print("mAP:"+ str(round(np.mean(avg_precision),4)*100) +"%" + " rank 1: "+ str(round(np.mean(np.array(r1)),4)*100) +"%" )
	print(avg_precision)

def generate_features(model=None, loader=None,save_path=None):

	features = np.zeros((0,2048))
	pid = []
	cam = []
	path = []

	# load from save if possible
	if save_path is not None and os.path.exists(save_path):
		print("Loading features from save: " + save_path)
		features,pid,cam,path =  pickle.load( open( save_path, "rb" ) )
	else:
		print("No save detected. Generating new features.")
		for i,samples in enumerate(loader):

			images, pids, cameras, paths = samples['image'], samples['pid'], samples['camera'], samples['path']

			forward = model.forward(images.cuda())[0]
			batch_size = images.shape[0]
			np_batch = forward.data.cpu().numpy()

			features = np.concatenate((features,np_batch),0)
			pid.extend(pids.data.cpu().numpy().tolist())
			cam.extend(cameras.data.cpu().numpy().tolist())
			path.extend(paths)

		if save_path is not None:
			print("Saving features...")
			pickle.dump( (features,pid,cam,path), open( save_path, "wb" ) )

	return features,pid,cam,path
	
def calculate_ap(binary_labels):
	
	ap = 0
	num_correct =0
	#Calculate AP@75
	for i in range(75):
		num_correct += binary_labels[i]
		ap += (num_correct/float(i+1)) * binary_labels[i] *(1.0/np.sum(binary_labels))
	return ap

def display(query,gallery,dataset_dir):

	gallery_path, gallery_pid, gallery_cam,gallery_dist = gallery

	command = "montage -label query_1503_cam_7 " + dataset_dir+"/query/"+query
	
	for i, image in enumerate(gallery_path):
		command += " -label \"gallery_" + str(gallery_pid[i]) + "_cam_" + str(str(gallery_cam[i])) + "\ndist_" + str(round(gallery_dist[i],3)) + "\" " + dataset_dir + "/bounding_box_test/" + image
	command += " output.jpg"
	print("Saved display output to output.jpg")
	os.system(command)

if __name__ == "__main__": main()
