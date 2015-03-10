import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cPickle as pickle


def get_data(file):
	'''
	INPUT: filepath to data in csv format
	OUTPUT: 3 pandas dataframes
		-continuous predictors
		-categorical predictors
		-outcome variables
	'''
	print 'loading data'
	df = pd.read_csv(file, delimiter='\t')
	condition = df.age >= 18
	df_sub = df[condition] #removing children and adolescents
	df_y = df_sub.pop('trt') #treatment is not part of the clustering
	cont_names = ['age', 'pga', 'bsa'] #continuous variables
	df_cont = df_sub[cont_names] 
	cat_names = ['sex', 'race', 'smoking', 'famhist'] 
	df_cat = df_sub[cat_names] #categorical variables
	return df_cont, df_cat, df_y

class Model(object):
	def __init__(self):
		self.model = KMeans()

	def _col_stats(self):
		'''
		Calculates the column statistics to be used for scaling the continuous
		variables in the new patient classification
		'''
    	names = df_cont.columns
    	means = df_cont.mean(axis=0)
    	stds = df_cont.std(axis=0)
    	self.cont_stats = {name : (means[name], stds[name]) for name in names}
 	
	def _preprocess(self):
		'''
		INPUT: two pandas dataframes
			-continuous variables
			-categorical variables
		OUTPUT: None, saving a numpy array to be used later
 		'''
		df_cont = (df_cont - df_cont.mean()) / df_cont.std()
		df_cat = pd.get_dummies(df_cat)
		self.X = pd.concat((df_cont, df_cont), axis=1).values

	def _optimal_k(self, max_k, n_iter=5):
		'''
		INPUT: 
			-numpy array of preprocessed predictor values
			-the max number of clusters
			-the number of random starts for each Kmeans algorithm
		OUTPUT: optimized k saved
		'''
		k_list = []
		score_list = []
		for k in xrange(1, max_k+1):
			score = []
			for i in range(n_iter):
				kmeans=self.model(n_clusters=k, init='random')
				kmeans.fit(self.X)
				score.append(kmeans.score(self.X))
			k_list.append(k)
			score_list.append(np.mean(score))
		######################################
		#need to find a way to select a k
		####################################
		self.k_list
		self.score_list
		self.best_k = 8

	def plot(self):
		return plt.plot(self.k_list, self.score_list)

	def final_fit(self, df_cont, df_cat, max_k):
		'''
		INPUT: 
			-the dataframe of continuous predictors
			-the dataframe of categorical predictors
			-the maximum number of clusters to check
		OUTPUT: the fitted model with optimized # of clusters
		'''
		print 'preprocessing'
		self._col_stats(df_cont)
		self._preprocess(df_cont, df_cat)

		print 'finding optimal number of clusters'
		self._optimal_k(max_k)

		print 'building final model'
		kmeans=self.model(n_clusters=self.best_k)
		self.final_fit = kmeans.fit(self.X)

	def classify_new_patient(self, input_values):
		'''
		INPUT: a dictionary of user input values for a new patient	
		OUTPUT: a cluster level
		'''
		input_dict = {'sex' : {'Female' : [1, 0], 'Male' : [0, 1]},
					  'race' : {'Asian' : [1, 0, 0, 0, 0], 'Black' : [0, 1, 0, 0, 0],
					            'Other' : [0, 0, 1, 0, 0], 'Uknown' : [0, 0, 0, 1, 0],
					            'White' : [0, 0, 0, 0, 1]},
					  'smoking' : {'Current Smoker' : [1, 0, 0, 0], 'Former Smoker' : [0, 1, 0, 0],
					               'Never Smoker' : [0, 0, 1, 0], 'Unknown' : [0, 0, 0, 1]},
					  'famhist' : {'No' : [1, 0], 'Yes' : [0, 1]}}
		new_input = []
		for var in self.cont_names:
			new_input.append((int(input_values[var]) - self.cont_stats[var][0]) / self.cont_stats[var][1])
		for var in self.cat_names:
			new_input = new_input + input_dict[var][input_values[var]]
		new_input = np.array(new_input)

def build_model(filepath):
	df_cont, df_cat, df_y = get_data(filepath)
	model = Model()
	fit = model.final_fit(df_cont, df_cat, max_k=10)
	scree_plot = self.plot()
	return fit, scree_plot

if __name__ == '__main__':
	filepath = '../../data/sample/analysis_dataset.tsv'
	model, scree_plot = build_model(filepath)
