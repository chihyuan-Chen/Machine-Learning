# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn import datasets
import numpy as np

import os
import json


os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["172.17.0.4:20001", "172.17.0.5:20001", "172.]
    },
    'task': {'type': 'worker', 'index': 2}
})


iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals1 = np.array([1 if y==0 else -1 for y in iris.target])
y_vals2 = np.array([1 if y==1 else -1 for y in iris.target])
y_vals3 = np.array([1 if y==2 else -1 for y in iris.target])
y_vals = np.array((y_vals1, y_vals2, y_vals3))
y_vals = np.transpose(y_vals)
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.7), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = np.array([y_vals1[train_indices], y_vals2[train_indices], y_vals3[train_indices]])
#y_vals_train = np.transpose(y_vals_train)
y_vals_test = np.array([y_vals1[test_indices], y_vals2[test_indices], y_vals3[test_indices]])
y_vals_test = np.transpose(y_vals_test)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(2,))
    model.add(tf.keras.layers.Dense(1,
                           kernel_regularizer=tf.keras.regularizers.l2(),
                           activation='linear'))
    
    '''
    <<meta paremeter>>
      1.AdamOptimizer(0.005)
      2.loss using Hinge, reduction sum over batch
    '''
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.005) ,   
        loss = tf.keras.losses.Hinge(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
    )

#show model
model.summary()


w = []
setosa_x= []
setosa_y = []
versicolor_x = []
versicolor_y = []
virginica_x = []
virginica_y = []
best_fit = []
best_fit1 = []
best_fit2 = []
x1_vals = []
for j in range(0,3):
#training
    model.fit(x_vals_train,y_vals_train[j][:],epochs = 500)

    #extract weight
    for layer in model.layers:
        w = layer.get_weights()


    #get coefficient
    A = w[0]
    b = w[1]

    #evaluate slope and intercept
    slope = -A[1]/A[0]
    y_intercept = b/A[0]

    #reload all data for showing graph
    ds = tfds.load('iris', split='train')
    x1_vals = [i['features'][3] for i in ds]


    for i in x1_vals:
        if j==0:
            best_fit.append(slope*i+y_intercept)
        elif j==1:
            best_fit1.append(slope*i+y_intercept)
        else:
            best_fit2.append(slope*i+y_intercept)
    if j==0:
        setosa_x = [i['features'][3] for i in ds if i['label'] == 0]
        setosa_y = [i['features'][0] for i in ds if i['label'] == 0]
    elif j==1:
        versicolor_x = [i['features'][3] for i in ds if i['label'] == 1]
        versicolor_y = [i['features'][0] for i in ds if i['label'] == 1]
    else:
        virginica_x = [i['features'][3] for i in ds if i['label'] == 2]
        virginica_y = [i['features'][0] for i in ds if i['label'] == 2]

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# Plot data and line
plt.plot(setosa_x, setosa_y, 'o', label='setosa')
plt.plot(versicolor_x, versicolor_y, 'x', label='versicolor')
plt.plot(virginica_x, virginica_y, '+', label='virginica')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=6)
plt.plot(x1_vals, best_fit1, 'g-', label='Linear Separator', linewidth=12)
plt.plot(x1_vals, best_fit2, 'b-', label='Linear Separator', linewidth=6)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()
plt.savefig('mid.jpg')