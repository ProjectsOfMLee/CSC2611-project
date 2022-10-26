from pyspark import SparkContext as SC
from pyspark import SparkConf 
import os


# initialize spark
conf = (SparkConf()
		.setMaster("local")
		.setAppName("SemanticProj") \
		#.set("spark.executor.memory", "1g") 
		)
sc = SC(conf = conf)

def extract_load_to_rdd(data=None, path=None, file_path=None):
	"""
	data: list of preprocessed words
	path: specific path to .txt
	file_path: path to the folder that holds all .txt
	"""
	if path:
		"""
		with open(path, 'r') as f:
    		data = f.read().split() # to list of strings for every txt file
		"""
		rdd = sc.textFile()

	if file_path:
		"""
		for txt_path in os.listdir(file_path):
			path = os.path.join(file_path, txt_path) 
			with open(path, 'r') as f:
				data = f.read().split()
		"""
		rdd = sc.wholeTextFiles(file_path)

	if data:
		rdd = sc.parallelize(data)

	else:
		print("no data to load, send in a list to data or specify path")

	return rdd if not rdd.isEmpty() else None


def eda_with_rdd(rdd):
	pass

def transform_rdd(rdd):
	pass