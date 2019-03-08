from huanghoujing import HuangHoujingPytorch
from layumi import LayumiPytorch
from darenet import DareNetPytorch
class ReIDModel():


	def __init__(self,version='huanghoujing',transform=None,preprocessor=None):
		
		self.model = None
		self.preprocessor = None
		self.transform = None
		
		if version == 'huanghoujing':
			self.model = HuangHoujingPytorch()
		elif version == 'layumi':
			self.model= LayumiPytorch()
		elif version == 'darenet':
			self.model = DareNetPytorch()
			
		self.transform = self.model.transform
		self.preprocessor = self.model.preprocessor
			


	def forward(self,x):
		return self.model.forward(x)
