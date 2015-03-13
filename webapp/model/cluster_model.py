import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import cPickle as pickle
import matplotlib.pyplot as plt
from collections import Counter
from bar_chart import barchart


def get_data(file):
    '''
    INPUT: filepath to data in csv format
    OUTPUT: 3 pandas dataframes
        -continuous variables
        -categorical variabless
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
    def __init__(self, df_cont, df_cat, df_y, max_k=10):
        self.df_cont = df_cont
        self.cont_names = df_cont.columns
        self.df_cat = df_cat
        self.cat_names = df_cat.columns
        self.df_y = df_y
        self.max_k = max_k
        self.cat_dict = {'sex' : {'Female' : [1, 0], 'Male' : [0, 1]},
                      'race' : {'Asian' : [1, 0, 0, 0, 0], 'Black' : [0, 1, 0, 0, 0],
                                'Other' : [0, 0, 1, 0, 0], 'Uknown' : [0, 0, 0, 1, 0],
                                'White' : [0, 0, 0, 0, 1]},
                      'smoking' : {'Current Smoker' : [1, 0, 0, 0], 'Former Smoker' : [0, 1, 0, 0],
                                   'Never Smoker' : [0, 0, 1, 0], 'Unknown' : [0, 0, 0, 1]},
                      'famhist' : {'No' : [1, 0], 'Yes' : [0, 1]}}

    def _col_stats(self, df):
        '''
        INPUT: takes in a continous dataframe
        OUTPUT: None, saves the column statistics to be used later
        '''
        names = df.columns
        means = df.mean(axis=0)
        stds = df.std(axis=0)
        self.cont_stats = {name : (means[name], stds[name]) for name in names}
    
    def _preprocess(self, df1, df2):
        '''
        INPUT: two pandas dataframes
            -continuous variables
            -categorical variables
        OUTPUT: None, saving a numpy array to be used later
        '''
        df1 = (df1 - df1.mean()) / df1.std()
        df2 = pd.get_dummies(df2)
        self.X = pd.concat((df1, df2), axis=1).values

    def _make_rand_data(self, df1, df2):
        '''
        INPUT: df1 is dataframe of continuous variables
               df2 is dataframe of categorical variables
        OUTPUT: an array of random values for the continuous variables and the equivalent
                set of indicator variables to identify each categorical variable 
        '''
        for name in self.cont_names:
            df1[name] = np.random.uniform(df1[name].min(), df1[name].max(), df1.shape[0])
        for name in self.cat_names:
            df2[name] = np.random.choice(self.cat_dict[name].keys(), df1.shape[0])
        return self._preprocess(df1, df2)

    def _gap_statistic(self, X, max_k, B=10):
        '''
        INPUT: takes in numpy array of data and maximum number of clusters to check
        OUTPUT: the optimal cluster size
        '''
        k_list = range(1, max_k+1)
        km_act_inertias = np.zeros(len(k_list))
        ref_inertias = np.zeros(len(k_list))
        sks = np.zeros(len(k_list))
        for i, k in enumerate(k_list):
            kmeans_model = KMeans(n_clusters=k)
            kmeans_model.fit(X)
            km_act_inertias[i] = np.log(kmeans_model.inertia_)
            #creating the reference datasets
            B_inertias = np.zeros(B)
            for b in range(B):
                ref = self._make_rand_data(self.df_cont, self.df_cat)
                kmeans_model = KMeans(n_clusters=k)
                kmeans_model.fit(ref)
                B_inertias[b] = np.log(kmeans_model.inertia_)
            ref_inertias[i] = np.sum(B_inertias)/float(B)
            #calculate the standard deviations
            sks[i] = np.std(B_inertias)*np.sqrt(1+(1./B))
        self.k_list = k_list
        self.gap = ref_inertias - km_act_inertias
        self.sks = sks
        self.criteria = self.gap[:-1] - (self.gap - self.sks)[1:]
        self.best_k = np.argmax(self.criteria) + 1

    def final_fit(self):
        '''
        INPUT: 
            -the dataframe of continuous variables
            -the dataframe of categorical variables
            -the maximum number of clusters to check
        OUTPUT: the fitted model with optimized # of clusters
        '''
        print 'preprocessing'
        self._col_stats(self.df_cont)
        self._preprocess(self.df_cont, self.df_cat)

        print 'finding optimal number of clusters'
        #self._gap_statistic(self.X, self.max_k)

        print 'building final model'
        self.best_k=8
        self.model=KMeans(n_clusters=self.best_k)
        self.model.fit(self.X)

        print 'exporting cluster level info including plots'
        self.cluster_level_info()
        return self

    def classify_new_patient(self, input_values):
        '''
        INPUT: a dictionary of user input values for a new patient  
        OUTPUT: an assigned cluster label for that patient
        '''
        new_input = []
        for var in self.cont_names:
            new_input.append((int(input_values[var]) - self.cont_stats[var][0]) / self.cont_stats[var][1])
        for var in self.cat_names:
            new_input = new_input + self.cat_dict[var][input_values[var]]
        new_input = np.array(new_input)
        return self.model.predict(new_input)

    def cluster_level_info(self):
        self.treatments = {treatment for treatment in self.df_y}
        cluster_results_dict = {}
        for cluster in range(self.model.n_clusters):
            c = Counter()
            for treatment in self.treatments:
                c[treatment] = 0
            for treatment in self.df_y[self.model.labels_ == cluster]:
                c[treatment] += 1
            cluster_results_dict[cluster] = {'treatments':c}
            plot_url = barchart(c, 5, cluster)
            cluster_results_dict[cluster].update({'bar_chart':plot_url})
        self.cluster_results_dict = cluster_results_dict

def build_model(filepath, model_filename):
    df_cont, df_cat, df_y = get_data(filepath)
    model = Model(df_cont, df_cat, df_y, max_k=10).final_fit()
    
    if model_filename:
        with open(model_filename, 'wb') as fp:
            pickle.dump(model, fp)
    return model

if __name__ == '__main__':
    filepath = '../../data/sample/analysis_dataset.tsv'
    model_filename = '../webapp/model.pkl'
    build_model(filepath, model_filename)
    












