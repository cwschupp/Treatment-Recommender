
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/clayton.schupp/cwschupp/project/data/sample/analysis_dataset.tsv', delimiter='\t')
df.head()
df.shape


df.sex.value_counts()
m=df[df.sex=='Male']
f=df[df.sex=='Female']

pd.crosstab(df.sex, df.pga)/df.pga.value_counts()

plt.figure(1)
plt.subplot(131)
df.bsa.hist()
plt.title('Overall')
plt.subplot(132)
m.bsa.hist()
plt.title('Males')
plt.subplot(133)
f.bsa.hist()
plt.title('Females')


df.race.value_counts()
pd.crosstab(df.race, df.pga)/df.pga.value_counts()


df.smoking.value_counts()
pd.crosstab(df.smoking, df.pga)/df.pga.value_counts()


df.famhist.value_counts()
pd.crosstab(df.famhist, df.pga)/df.pga.value_counts()


df.pga.value_counts()
pga2 = df[df.pga==2]
pga3 = df[df.pga==3]
pga4 = df[df.pga==4]
pga5 = df[df.pga==5]
pga6 = df[df.pga==5]


plt.figure(2)
plt.subplot(231)
df.bsa.hist()
plt.title('Overall')
plt.subplot(232)
pga2.bsa.hist()
plt.title('2')
plt.subplot(233)
pga3.bsa.hist()
plt.title('3')
plt.subplot(234)
pga4.bsa.hist()
plt.title('4')
plt.subplot(235)
pga5.bsa.hist()
plt.title('5')
plt.subplot(236)
pga6.bsa.hist()
plt.title('6')





