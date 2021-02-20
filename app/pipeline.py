import joblib
from sklearn.pipeline import Pipeline

class Model_Pipeline:
	def __init__(self,  file):

		self.log = logg.get_class_log(self)
		self.pipeline = joblib.load(args)

	def pipelineData(self, data):
		return self.pipeline.transform(data)







