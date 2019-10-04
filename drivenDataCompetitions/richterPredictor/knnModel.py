# Applying Nearest neighbors model with different number of neighbors and
# plotting the Accuracy for each model.

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time

start = time.time()

X = pd.read_csv('noScalingPPTrainValues.csv', index_col='building_id')
y = pd.read_csv('train_labels.csv', index_col='building_id')

trainX, testX, trainy, testy = train_test_split(X, y, stratify=y)

numNeighbors = list(range(5, 51, 5))
accuracies = []

for numNei in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=numNei)
    clf.fit(trainX, trainy.values.ravel())
    yPred = clf.predict(testX)
    accuracies.append(f1_score(testy, yPred, average='micro'))

plt.plot(numNeighbors, accuracies)
plt.gca().set(xlabel='Number of neighbors', ylabel='Accuracy',
              title='Accuracy for different number of neighbors')
plt.show()

print('time : ' + str((time.time() - start)/60))
