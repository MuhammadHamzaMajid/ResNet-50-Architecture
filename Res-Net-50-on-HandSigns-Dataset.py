import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Add, Dense, ReLU, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
#1. Data Pre-processing
#a. load the dataset
path = "E:\\Deep-Learning-Specialization\\Convolutional-Neural-Networks\\ResNet-50-Architecture\\DATASET"
#b. determine image dimensions
img_height, img_width = 64, 64
#c. prepare the lists
X = []; Y = []
class_names = sorted(os.listdir(path))
for label, class_name in enumerate(class_names):
    class_folder = os.path.join(path, class_name)
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        img = load_img(img_path, target_size = (img_height, img_width))
        img_array = img_to_array(img)
        X.append(img_array)
        Y.append(label)
#d. convert list to array and normalize the images
X= np.array(X, dtype= "float32") / 255.
Y = np.array(Y)
#e. split the data into train/test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
#f. convert labels to one-hot encoding
num_classes = len(class_names)
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)


#The Identity Block of a ResNet
'''
x --> |Conv2D|BatchNorm|ReLU| --> |Conv2D|BatchNorm|ReLU| --> |Conv2D|BatchNorm| --> |+| --> |ReLU| --> output
|                                                                                     |
>-------------------------------------x_shortcut--------------------------------------|

'''
def identity_block(X, f, filters, initializer = random_uniform):
    '''
    X = input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f = integer, for the middle Conv2D layer's filter size
    filters = a list of no. of filters for every Conv2D layer on the main path
    '''
    #Retrieve filters
    F1, F2, F3 = filters
    #Save the input value
    X_shortcut = X

    #First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1, 1), padding = "valid", kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X) # the channels axis
    X = ReLU()(X)

    #Second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding = "same", kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)
    X = ReLU()(X)

    #Third component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding = "valid", kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)

    #Final Step: Add the Shortcut Value to the main path, add pass it through a ReLU activation
    X = Add()([X, X_shortcut])
    X = ReLU()(X)
    return X

#The Convolutional Block of a ResNet
'''
x --> |Conv2D|BatchNorm|ReLU| --> |Conv2D|BatchNorm|ReLU| --> |Conv2D|BatchNorm| --> |+| --> |ReLU| --> output
|                                                                                     |
>--------------------------------- > |Conv2D|BatchNorm| ------------------------------|
'''
def convolutional_block(X, f, filters, s = 2, initializer = glorot_uniform):
    #Retrieve filters
    F1, F2, F3 = filters
    #Save the input value for the shortcut path
    X_shortcut = X

    #First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding = 'valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)
    X = ReLU()(X)

    #Second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding = 'same', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)
    X = ReLU()(X)

    #Third Component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding = 'valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)

    #Shortcut Path
    X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding = 'valid', kernel_initializer = initializer())(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    #Final Step: Add shortcut to main path
    X = Add()([X, X_shortcut])
    X = ReLU()(X)

    return X

#The ResNet-50 Model
'''
input --> |ZeroPad2D| --> |Conv2D|BatchNorm|ReLU|MaxPool2D| --> |ConvBlock|IDBlock x2| --> |ConvBlock|IDBlock x3| --> |ConvBlock|IDBlock x5| --> |ConvBlock|IDBlock x2| --> |AvgPool|Flatten|FC| --> Output
                                         Stage1                         Stage2                     Stage3                     Stage4                     Stage5                    Stage6
'''
def ResNet50(input_shape = (64, 64, 3), classes = 6, training = False):
    #Define input as a tensor
    X_input= Input(input_shape)

    #ZeroPadding
    X = ZeroPadding2D((3, 3))(X_input)

    #Stage 1
    X = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = "same", kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3)(X)
    X = ReLU()(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)

    #Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, f = 3, filters = [64, 64, 256])
    X = identity_block(X, f = 3, filters = [64, 64, 256])

    #Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    X = identity_block(X, f = 3, filters = [128, 128, 512])
    X = identity_block(X, f = 3, filters = [128, 128, 512])
    X = identity_block(X, f = 3, filters = [128, 128, 512])

    #Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    
    #Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, f = 3, filters = [512, 512, 2048])
    X = identity_block(X, f = 3, filters = [512, 512, 2048])

    #AveragePooling
    X = AveragePooling2D(pool_size = (2, 2))(X)
    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax', kernel_initializer = glorot_uniform())(X)

    #create model
    model = Model(inputs = X_input, outputs = X)

    return model

#Get the model shape
model = ResNet50(input_shape = (64, 64, 3), classes = num_classes)
print(model.summary())

#Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00015), loss = "categorical_crossentropy", metrics = ["accuracy"])

#Fit the model on the training set
history = model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

#Plotting the training accuracy
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()