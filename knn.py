#load irisdata
from sklearn.datasets import load_iris
iris = load_iris()
#store x & y
X = iris.data
y = iris.target
#split data into test & train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,random_state = 1 )
#training model on train dataset
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) 

#making predictions on the testing set
y_pred = knn.predict(X_test)
#comparing response ytest with predictd y
from sklearn import metrics
print("Knn model accuracy:", metrics.accuracy_score(y_test, y_pred))

#making predictions for output of sample data
sample = [[3, 5, 4, 2], [2, 3, 5, 4]]
preds = knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds]
print("Predictions:", pred_species)

#saving the model
from sklearn.externals import joblib
joblib.dump(knn, 'iris_knn.pkl')

#output
#kNN model accuracy: 0.983333333333
#Predictions: ['versicolor', 'virginica']
