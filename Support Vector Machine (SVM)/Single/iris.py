# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn import datasets
import numpy as np

'''
<<data preprocessing>>
  1.load 'iris' dataset
  2.spliting dataset in to 'train' & 'test' which is 80-20
  3.
    (1).select ['Sepal Length', 'Petal Width'] as features
    (2).label on select 'setosa' be 1, other be 0
'''
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]

iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y==0 else 2 if y==1 else 3 for y in iris.target])
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.7), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


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

#model_to_estimator
strategy = None
#strategy = tf.distribute.MultiWorkerMirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)

estimator = tf.keras.estimator.model_to_estimator(model, config=config)


#training

def input_fn(x_vals, y_vals, epochs, batch_size):
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_vals, y_vals))

    # Shuffle, repeat, and batch the examples. 
    SHUFFLE_SIZE = 50
    dataset = dataset.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    dataset = dataset.prefetch(None)

    # Return the dataset. 
    return dataset

BATCH_SIZE=32
EPOCHS = 1000
estimator_train_result = estimator.train(input_fn=lambda:input_fn(x_vals_train, y_vals_train, epochs=EPOCHS, batch_size=BATCH_SIZE))

#find variable
name = estimator_train_result.get_variable_names()
print(estimator_train_result.get_variable_names())
for i in name:
    print(i,":",estimator_train_result.get_variable_value(i))


A = estimator_train_result.get_variable_value(name[5])  
b = -estimator_train_result.get_variable_value(name[2])

#evaluate slope and intercept
slope = -A[1]/A[0]
y_intercept = b/A[0]

#reload all data for showing graph
ds = tfds.load('iris', split='train')
x1_vals = [i['features'][3] for i in ds]*3
best_fit=[]
for i in x1_vals:
    best_fit.append(slope*i+y_intercept)

setosa_x = [i['features'][3] for i in ds if i['label'] == 0]
setosa_y = [i['features'][0] for i in ds if i['label'] == 0]
versicolor_x = [i['features'][3] for i in ds if i['label'] == 1]
versicolor_y = [i['features'][0] for i in ds if i['label'] == 1]
virginica_x = [i['features'][3] for i in ds if i['label'] == 2]
virginica_y = [i['features'][0] for i in ds if i['label'] == 2]

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# Plot data and line
x1_vals = np.array(x1_vals)
best_fit = np.array(best_fit)
plt.plot(setosa_x, setosa_y, 'o', label='setosa')
plt.plot(versicolor_x, versicolor_y, 'x', label='versicolor')
plt.plot(virginica_x, virginica_y, '+', label='virginica')
plt.plot(x1_vals[:150], best_fit[:150], 'r-', label='Linear Separator', linewidth=3)
plt.plot(x1_vals[150:300], best_fit[150:300], 'g-', label='Linear Separator', linewidth=3)
plt.plot(x1_vals[300:450], best_fit[300:450], 'b-', label='Linear Separator', linewidth=3)
#plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()
