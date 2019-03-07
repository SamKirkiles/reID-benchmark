# ReID Benchmarking Environment
A general framework for benchmarking re-identification models.

![Market1501](http://www.liangzheng.org/Project/dataset.jpg)

#### Current Models Supported
* Aligned ReID *(huanghoujing)*
  mAP: 78.34% Rank1: 90.56%

#### Current Datasets Supported
* Market1501 [[Direct Link](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip)]
* CUHK-03 [[Google Drive](https://drive.google.com/file/d/1pBCIAGSZ81pgvqjC-lUHtl0OYV1icgkz/view)] 

Unzip each datset into the `/data` directory. Datasets must be in Market1501 format to be used in this package. 

### Getting Started
#### Install Dependenceies
Run `pip install -r requirements.txt` to download necessary dependencies. It is recomended to use virtualenv to manage dependencies. 
#### Download Datasets
Unzip your desired dataset using the download links provided above into the `/datasets` folder. If using a dataset from a source other than the links above it must be converted to the [Market1501 format](http://www.liangzheng.org/Project/project_reid.html).
#### Clone ReID Model Into `/models`
To start benchmarking, a model and associated wrapper class must be provided. Note that only models written in Python 3 are supported. 
#### Create Wrapper Class
#### Run Benchmark
