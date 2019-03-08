# ReID Benchmarking Environment
A general framework for benchmarking re-identification models.

![Market1501](http://www.liangzheng.org/Project/dataset.jpg)

### Current Models Supported
#### Aligned ReID *(huanghoujing)*  [[Repo](https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch)]
   ###### Trained on Market1501
| Benchmark    | Market1501   |Cuhk03  |
| ------------- |:-------------:| :-----:|
| Aligned ReID without re-ranking| **mAP:** 78.34% **rank1:** 90.56%| **mAP:** - **rank 1:** - |
| Aligned ReID with re-ranking| **mAP:** 89.37% **rank 1:** 93.02% | **mAP:** 17.47% **rank 1:** 15.21% |

#### Person_reID_baseline_pytorch *(layumi)* [[Repo](https://github.com/layumi/Person_reID_baseline_pytorch)]
   ###### Trained on Market1501
| Benchmark    | Market1501   |Cuhk03  |
| ------------- |:-------------:| :-----:|
| ReID without re-ranking| **mAP:** - **rank1:** -| **mAP:** - **rank 1:** - |
| ReID with re-ranking| **mAP:** 86.59% **rank 1:** 92.07% | **mAP:** 18.98% **rank 1:** 17.43% |

#### DareNet *(mileyan)* [[Repo](https://github.com/mileyan/DARENet)]
   ###### Trained on Market1501
| Benchmark    | Market1501   |Cuhk03  |
| ------------- |:-------------:| :-----:|
| ReID without re-ranking| **mAP:** - **rank1:** -| **mAP:** - **rank 1:** - |
| ReID with re-ranking| **mAP:** 86.02% **rank 1:** 90.5% | **mAP:** 10.29% **rank 1:** 8.70% |

   ###### Trained on Cuhk03
   | Benchmark    | Market1501   |Cuhk03  |
| ------------- |:-------------:| :-----:|
| ReID without re-ranking| **mAP:** - **rank1:** -| **mAP:** - **rank 1:** - |
| ReID with re-ranking| **mAP:** - **rank 1:** - | **mAP:** 78.46% **rank 1:** 75.79% |


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
