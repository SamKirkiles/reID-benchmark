from reid_model import ReIDModel
import argparse
import datasets

def evaluate(args):
	print("Starting evaluation")

	model = ReIDModel(version='huanghoujing')

	dataset = datasets.create('market1501', 'data/{}'.format('market1501'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Evaluate")
	parser.add_argument('-b', '--batchsize', type=int, default=50)
    evaluate(parser.parse_args())
