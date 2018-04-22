import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('Heart.csv')
data = data.drop(data.columns[[0]],axis = 1)
data['AHD'] = data['AHD'].map({"Yes" :1 , "No" : -1})
null = list(data.columns[data.isnull().any()])
data['Ca'] = data['Ca'].fillna(data['Ca'].median())
#print data.Thal.value_counts()
data.Thal = data.Thal.fillna("normal")
vect = ['ChestPain','Thal']
for i in vect:
        temp = pd.get_dummies(pd.Series(data[i]))
        data = pd.concat([data,temp],axis = 1)
        data = data.drop([i],axis = 1)
y = data['AHD']
del data['AHD']

data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.3,random_state=0)


# Initialize D
D = [1.0/len(data_train)]*len(data_train)
     
T = 10 #Max number of iteration

a = []
yout_train = []
yout_test = []
for i in range(T):
        s = svm.SVC()
        s.fit(data_train,y_train,sample_weight = D)
        y_out_train = s.predict(data_train)
        y_out_test = s.predict(data_test)
        ''' 
        t = tree.DecisionTreeClassifier()
        t.fit(data_train,y_train,sample_weight = D)
        y_out_train = t.predict(data_train)   
        y_out_test = t.predict(data_test)
        '''
        '''
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(data_train,y_train)
        y_out_train = neigh.score(data_train,y_train,sample_weight = D)
        y_out_test = neigh.predict(data_test)
        '''
        print ('acc_train',accuracy_score(y_train,y_out_train))
        print ('acc',accuracy_score(y_test,y_out_test))
        yout_train.append(y_out_train)
        yout_test.append(y_out_test)
        epslon = []
        for i in range(len(y_train)):
                if (y_train.iloc[i] != y_out_train[i]):
                        epslon.append(D[i])
        #print epslon
        #print 'y_train',y_train
        #print 'y_out_train',y_out_train
        a1 = 0.5*np.log((1-sum(epslon))/sum(epslon))  
        a.append(a1)
        Z = []
        for i in range(len(data_train)):
                z = D[i]*np.exp(-a1*y_train.iloc[i]*y_out_train[i])
                #print z
                Z.append(z) 
        Z = sum(Z) 
        for i in range(len(D)):
                D[i] = (D[i]*np.exp(-a1*y_train.iloc[i]*y_out_train[i])) / Z
        #print 'D_end',D

ada_out = []
for i in range(len(data_test)):
        ada = []
        for j in range(T):
                ada.append(a[j]*yout_test[j][i])
        ada = sum(ada)
        if (ada < 0):
                ada_out.append(-1)
        elif(ada > 0):
                ada_out.append(1)
print (accuracy_score(y_test,ada_out))
