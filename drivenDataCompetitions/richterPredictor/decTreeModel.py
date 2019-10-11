# Applying Decision Tree model using GridSearchCV. Also the part ********** of
# the code is computing accruacy of models with different max_depth and fixed
# min_samples_leaf.

import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

start = time.time()

X = pd.read_csv('noScalingPPTrainValues.csv', index_col='building_id')
y = pd.read_csv('train_labels.csv', index_col='building_id')

trainX, testX, trainy, testy = train_test_split(X, y, stratify=y,
                                                random_state=12)

# Make this line a comment when running part **********.

params = {'max_depth': [i for i in range(25, 35)], 'min_samples_leaf':
          [i for i in range(30, 40)]}


# **********
# Setting 'min_samples_leaf=35', this piece of code finds the accuracy for
# different max_depth parameter.

# acc = []
# maxDepthList = [i for i in range(20, 40)]
# for i in maxDepthList:
#     dtc = DecisionTreeClassifier(max_depth=i, random_state=12,
#                                  min_samples_leaf=35)
#     dtc.fit(trainX, trainy)
#     predy = dtc.predict(testX)
#     acc.append(f1_score(testy, predy, average='micro'))


# Make comment this piece of code when running part **********.

dtc = DecisionTreeClassifier(random_state=12)
gSearcher = GridSearchCV(dtc, param_grid=params, scoring='f1_micro', cv=5)
gSearcher.fit(trainX, trainy.values.ravel())
print('Best hyperparameters : \n', gSearcher.best_params_)
print('Best score : ', gSearcher.best_score_)
predy = gSearcher.predict(testX)
print('Test score : ', f1_score(testy, predy, average='micro'))


# Plotting the accuracy of the models created in part **********.

# plt.plot(maxDepthList, acc, linestyle='-', marker='.', markersize=8)
# plt.gca().set(xlabel='Maximum depth', ylabel='Accuracy',
#               title='Accuracy of each model', xticks=maxDepthList,
#               xlim=(min(maxDepthList), max(maxDepthList)))
# plt.grid(linestyle='--')
# plt.show()


print('time : ' + str((time.time() - start)/60))
