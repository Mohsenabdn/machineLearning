from sklearn.neighbors import KNeighborsClassifier
#from ordinalClassifier import OrdinalClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
import time

start = time.time()
X = pd.read_csv('noScalingPPTrainValues.csv', index_col='building_id')
y = pd.read_csv('train_labels.csv', index_col='building_id')
# pca = PCA()
# X = pca.fit_transform(X)
trainX, testX, trainy, testy = train_test_split(X, y, stratify=y)
numNeighbors = list(range(5, 101, 5))
accuracies = []

for numNei in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=numNei)
    clf.fit(trainX, trainy.values.ravel())
    #print(clf.score(testX, testy))
    yPred = clf.predict(testX)
    accuracies.append(f1_score(testy, yPred, average='micro'))

plt.plot(numNeighbors, accuracies)
plt.gca().set(xlabel='Number of neighbors', ylabel='Accuracy', title='Accuracy for different number of neighbors')
plt.show()
print('time : ' + str((time.time() - start)/60))
