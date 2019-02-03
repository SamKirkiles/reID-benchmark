from huanghoujing import HuangHoujingPytorch

class ReIDModel():


	def __init__(self,version='huanghoujing'):
		
		self.model = None

		if version == 'huanghoujing':
			self.model = HuangHoujingPytorch()


	def forward(self,x):
		return self.model.forward(x)
