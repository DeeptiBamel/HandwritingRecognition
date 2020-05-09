#Dataset - MNIST which has items of handwriting -- the digits 0 through 9.
#MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs 
#-- i.e. you should stop training once you reach that level of accuracy.

import tensorflow as tf
from os import path, getcwd, chdir

# CHANGE THE LINE BELOW - edit the path to that location where dataset is present
path = f"{getcwd()}/../tmp2/mnist.npz"

# GRADED FUNCTION: train_mnist
def train_mnist():

    #callBack class definition - to stop next iteration once training reaches 99% accuracy
    class myCallBack(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.99):
                print("Reached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    
    #handwriting dataset is directly available in tf.keras API
    mnist = tf.keras.datasets.mnist

    #load is called on mnist object, will return 2 set of 2 lists - training and testing values for handwriting images and their labels
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)

    #here, we are normalising value from 0-255 to 0-1
    x_train, x_test = x_train / 255.0, x_test/255.0
    callbacks = myCallBack()

    #model designing
    #Sequential - sequence of layers in NN
    #Flatten - to transform image from square to 1-D set
    #Dense - layer of neurons and each one of these need ACTIVATION function
    #ReLu - functions as "if x>0 then return x, else return 0"
    #Softmax - if we have set of values then it assigns 1 to biggest value and 0 to others
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),		#input layer
        tf.keras.layers.Dense(512, activation=tf.nn.relu),	#hidden layer
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)	#output layer
    ])

    #compiling model with optimization and loss functions
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting - fitting training data to training labels so that it can figure out a relationship between them
    # and the model can be used for future predictions
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    
    return history.epoch, history.history['acc'][-1]
	
train_mnist()