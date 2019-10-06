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

trainX, testX, trainy, testy = train_test_split(X, y, stratify=y,
                                                random_state=12)

numNeighbors = list(range(10, 26))
accuracies = []

for k in numNeighbors:
    clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
    clf.fit(trainX, trainy.values.ravel())
    yPred = clf.predict(testX)
    accuracies.append(f1_score(testy, yPred, average='micro'))

plt.plot(numNeighbors, accuracies, linestyle='-', marker='.', markersize=8)
plt.gca().set(xlabel='Number of neighbors', ylabel='Accuracy',
              title='Accuracy of each Model', xticks=numNeighbors,
              xlim=(min(numNeighbors), max(numNeighbors)))
plt.grid(linestyle='--')
plt.show()

print('time : ' + str((time.time() - start)/60))
