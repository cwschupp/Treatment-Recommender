import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

def get_data(file):
	df = pd.read_csv(file, delimiter='\t')
	condition = df.age >= 18
	df_sub = df[condition] #removing children and adolescents
	df_y = df_sub.pop('trt') #treatment is not part of the clustering
	self.cont_names = ['age', 'pga', 'bsa'] #continuous variables
	df_cont = df_sub[self.cont_names] 
	self.cat_names = ['sex', 'race', 'smoking', 'famhist'] 
	df_cat = df_sub[self.cat_names] #categorical variables
	return df_cont, df_cat, df_y


class Model(object):
	def __init__(self):
		pass

    def _col_stats(self, df_cont):
    	names = df_cont.columns
    	means = df_cont.mean(axis=0)
        stds = df_cont.std(axis=0)
        #need these to scale the new patient information before passing into model
    	self.cont_stats = {name : (means[name], stds[name]) for name in names}
 
	def _preprocess(self, df_cont, df_cat):
		df_cont = (df_cont - df_cont.mean()) / df_cont.std()
		df_cont = pd.get_dummies(df_cont)
		return pd.concat(df_cont, df_cont, axis=1)


	def fit(self, df):
		model 

	def classify_new_patient(self, input_values):
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
	


if __name__ == '__main__':
	filepath = '../../data/sample/analysis_dataset.tsv'
	build_model(filepath)
