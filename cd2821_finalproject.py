
# coding: utf-8

# # Deep Learning for Computer Vision:  Final Project

# ### Computer Science: COMS W 4995 006

# #### Author: Carmem Domingues

# # Problem: Identifying plastic pollution in underwater ocean images

# Plastic Pollution in our waterways (rivers, lakes, oceans, etc.) is a huge environmental problem in today's world. It is predicted that by 2050 there will be more plastic (by weight) than fish in the ocean. 
# 
# I would like to use Deep Learning and Computer Vision to help solve this problem. The applications of this are many, but we start with the simple problem of using DL and CV to distinguish between underwater ocean images that contain pieces of plastic, and those that do not (and only contain normal marine life). 
# 
# Data was collected via Google Searches and curated manually. More information on the data collection process can be found on the Final Presentation and Final Report for this project.
# 
# This project is divided into two main Deep Learning and Computer Vision tasks (besides data collection): Image Classification and Object Detection and Classification
# 
# For the Image Classification part:
# 
# 1. Start with a simple ConvNet, to help us distinguish between plastic and no-plastic images
# 2. Use a ConvNet (VGG16) that was pre-trained on ImageNet to see how we do (modify the top layers for the binary case, freeze all layers up to the top, and retrain just the top ones) 
# 3. Unfreeze all layers and continue fine tuning
# 
# For the Object Detection part:
# 
# Try and identify which parts of the images are plastic vs. normal marine life (or humans). 
# 
# Note: Steps 1-3 are inspired by and leverage code from homework 5.

# In[1]:


import os
import h5py

import matplotlib.pyplot as plt
import time, pickle, pandas

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers

get_ipython().magic('matplotlib inline')


# In[2]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Setup some basics for the models:

# In[3]:


nb_classes = 2
class_name = {
    0: 'noplastic',
    1: 'plastic',
}


# In[4]:


# dimensions of our images
img_width, img_height = 150, 150
# Data location and metadata
train_data_dir = '../Images/data_for_modeling/train'
validation_data_dir = '../Images/data_for_modeling/validation'
nb_train_samples = 900+700 # should be 893 + 699
nb_validation_samples = 77+70
batch_size = 32
steps_per_epoch_train = nb_train_samples / batch_size
steps_per_epoch_val = nb_validation_samples / batch_size


# Use the code from homework 5 for showing the images:

# In[5]:


def show_sample(X, y, prediction=-1):
    """This function is for showing sample images"""
    im = X
    plt.imshow(im)
    if prediction >= 0:
        plt.title("Class = %s, Predict = %s" % (class_name[y], class_name[prediction]))
    else:
        plt.title("Class = %s" % (class_name[y]))

    plt.axis('on')
    plt.show()


# # Image Classification

# In[6]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')


# In[7]:


nb_train_samples, nb_validation_samples


# Show some sample images from the dataset - borrow code from homework 5

# In[8]:


for X_batch, Y_batch in validation_generator:
    for i in range(len(Y_batch)):
        show_sample(X_batch[i, :, :, :], Y_batch[i])
    break


# ## 1. Simple ConvNet

# ##### Define the model

# In[9]:


model_scn = Sequential()
model_scn.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model_scn.add(Activation('relu'))
model_scn.add(MaxPooling2D(pool_size=(2, 2)))

model_scn.add(Conv2D(32, (3, 3)))
model_scn.add(Activation('relu'))
model_scn.add(MaxPooling2D(pool_size=(2, 2)))

model_scn.add(Conv2D(63, (3, 3)))
model_scn.add(Activation('relu'))
model_scn.add(MaxPooling2D(pool_size=(2, 2)))

model_scn.add(Flatten())
model_scn.add(Dense(64))
model_scn.add(Activation('relu'))
model_scn.add(Dropout(0.5))
model_scn.add(Dense(1))
model_scn.add(Activation('sigmoid'))


# In[10]:


model_scn.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics=['accuracy'])

print(model_scn.summary())


# ##### Make some directories for storing logs

# In[11]:


get_ipython().magic('pushd')
get_ipython().magic('mkdir -p history')
get_ipython().magic('mkdir -p models')
get_ipython().magic('mkdir -p logs')
get_ipython().magic('cd logs')
get_ipython().magic('mkdir -p ./scn')
get_ipython().magic('mkdir -p ./vgg16_fine_tuning')
get_ipython().magic('popd')


# In[12]:


tensorboard_callback_scn = TensorBoard(log_dir='./logs/scn/', 
                                       histogram_freq=0, write_graph=True, write_images=False)
checkpoint_callback_scn = ModelCheckpoint('./models/scn_weights.{epoch:02d}-{val_acc:.2f}.hdf5', 
                                          monitor='val_acc', verbose=0, save_best_only=True, 
                                          save_weights_only=False, mode='auto', period=1)


# ##### Train the model

# In[13]:


get_ipython().run_cell_magic('time', '', 'nb_epoch_scn = 10\n\nhist_scn = model_scn.fit_generator(train_generator, \n              initial_epoch=0, \n              verbose=1, \n              validation_data=validation_generator, \n              steps_per_epoch=steps_per_epoch_train, \n              epochs=nb_epoch_scn, \n              callbacks=[tensorboard_callback_scn, checkpoint_callback_scn],\n              validation_steps=steps_per_epoch_val)\n                                                                                                                                   \npandas.DataFrame(hist_scn.history).to_csv("./history/scn.csv")')


# ##### Grab some validation batches and evaluate our accuracy:

# In[14]:


accuracies_scn = np.array([])
losses_scn = np.array([])

i=0
for X_batch, Y_batch in validation_generator:
    loss, accuracy = model_scn.evaluate(X_batch, Y_batch, verbose=0)
    losses_scn = np.append(losses_scn, loss)
    accuracies_scn = np.append(accuracies_scn, accuracy)
    i += 1
    if i == 20:
       break
       
print("Validation: accuracy = %f  ;  loss = %f" % (np.mean(accuracies_scn), np.mean(losses_scn)))


# In[15]:


X_test, y_test = next(validation_generator)
predictions_scn = model_scn.predict_classes(X_test, batch_size=32, verbose=0)


# ##### Visualize the predicted labels on a batch

# In[16]:


for i in range(32):
    show_sample(X_test[i, :, :, :], y_test[i], prediction=predictions_scn[i, 0])


# Most of the mis-predictions were true label "plastic", but predicted "no plastic" -> seems like there is a bias in the model

# Accuracy in this initial model was 70.7% - considerable, given the small dataset size (compare to the 10k of each cat/dog label in the cat/dog exercise we did).
# 
# However, we can likely do better with the following models.

# ## 2. Use a ConvNet (VGG16) that was pre-trained on ImageNet to see how we do

# Rather than training it from scratch, load in pre-trained weights, from a model trained on ImageNet. Then replace the last layer with one that fits our two class plastic vs. no plastic problem. 
# 
# Freeze the bottom layers of the network and only train the weights of the new last layers. 
# 
# This training produces a classifier that has 84% accuracy, which is quite an improvement over the last one. 

# ##### Build the VGG16 model - we need this since we will train some layers, rather than just loading in pre-trained weights and using them for prediction on my dataset

# In[17]:


def build_vgg16(framework='tf'):
    """This function creates the VGG16 architecture model""""

    if framework == 'th':
        # build the VGG16 network in Theano weight ordering mode
        backend.set_image_dim_ordering('th')
    else:
        # build the VGG16 network in Tensorflow weight ordering mode
        backend.set_image_dim_ordering('tf')
        
    model = Sequential()
    if framework == 'th':
        model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
        
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    return model


# In[21]:


# path to the model weights files.
weights_path = '../weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# Pulled weights from (googled for them): https://github.com/MinerKasch/applied_deep_learning/blob
    # /master/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
tf_model = build_vgg16('tf')
tf_model.load_weights(weights_path)


# In[22]:


tf_model


# ##### Add in the top layers as described above

# In[23]:


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
# print(Flatten(input_shape=tf_model.output_shape[1:]))
top_model.add(Flatten(input_shape=tf_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
print(tf_model.summary())
print(top_model.summary())


# ##### Add this model to the top of the VGG16 network, freeze all the weights except the top, and compile.

# In[24]:


# add the model on top of the convolutional base
tf_model.add(top_model)


# In[25]:


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in tf_model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
tf_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[26]:


tf_model.layers[0].trainable


# ##### Train top model with frozen layers

# In[27]:


tensorboard_callback_tt = TensorBoard(log_dir='./logs/train_top/', histogram_freq=0, write_graph=True, 
                                   write_images=False)
checkpoint_callback_tt = ModelCheckpoint('./models/train_top_weights.{epoch:02d}-{val_acc:.2f}.hdf5', 
                                      monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, 
                                      mode='auto', period=1)


# In[28]:


get_ipython().run_cell_magic('time', '', 'nb_epoch = 10\n\nhist_train_top = tf_model.fit_generator(train_generator, \n              initial_epoch=0, \n              verbose=1, \n              validation_data=validation_generator, \n              steps_per_epoch=steps_per_epoch_train, \n              epochs=nb_epoch, \n              callbacks=[tensorboard_callback_tt, checkpoint_callback_tt],\n              validation_steps=steps_per_epoch_val)\n                                                                                                                                   \npandas.DataFrame(hist_train_top.history).to_csv("./history/train_top.csv")')


# ##### Evaluate the model

# In[29]:


accuracies_train_top = np.array([])
losses_train_top = np.array([])

i=0
for X_batch, Y_batch in validation_generator:
    loss, accuracy = tf_model.evaluate(X_batch, Y_batch, verbose=0)
    losses_train_top = np.append(losses_train_top, loss)
    accuracies_train_top = np.append(accuracies_train_top, accuracy)
    i += 1
    if i == 20:
       break
       
print("Validation train top: accuracy = %f  ;  loss = %f" % (np.mean(accuracies_train_top), 
                                                             np.mean(losses_train_top)))


# 84% accuracy - not bad, but it can do better!

# ##### Show some sample predictions

# In[30]:


predictions_train_top = tf_model.predict_classes(X_test, batch_size=32, verbose=0)
for i in range(32):
    show_sample(X_test[i, :, :, :], y_test[i], prediction=predictions_train_top[i, 0])


# ## 3. Unfreeze all layers 

# Unfreeze the lower layers

# In[31]:


# set the first 25 layers (up to the last conv block)
# to trainable (weights will be updated)
for layer in tf_model.layers[:25]:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
tf_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[32]:


tf_model.layers[0].trainable


# ##### Train model

# In[33]:


tensorboard_callback_tt_unfreeze = TensorBoard(log_dir='./logs/train_top_unfreeze/', histogram_freq=0, 
                                               write_graph=True, write_images=False)
checkpoint_callback_tt_unfreeze = ModelCheckpoint(
                                './models/train_top_unfreeze_weights.{epoch:02d}-{val_acc:.2f}.hdf5', 
                                monitor='val_acc', verbose=0, save_best_only=True, 
                                save_weights_only=False, mode='auto', period=1)

nb_epoch_tt_unfreeze = 10

hist_train_top_unfreeze = tf_model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epoch_tt_unfreeze, 
              callbacks=[tensorboard_callback_tt_unfreeze, checkpoint_callback_tt_unfreeze],
              validation_steps=steps_per_epoch_val)
                                                                                                                                   
pandas.DataFrame(hist_train_top_unfreeze.history).to_csv("./history/train_top_unfreeze.csv")


# ##### Evaluate the model and show some predictions

# In[34]:


accuracies_train_top_unfreeze = np.array([])
losses_train_top_unfreeze = np.array([])

i=0
for X_batch, Y_batch in validation_generator:
    loss, accuracy = tf_model.evaluate(X_batch, Y_batch, verbose=0)
    losses_train_top_unfreeze = np.append(losses_train_top_unfreeze, loss)
    accuracies_train_top_unfreeze = np.append(accuracies_train_top_unfreeze, accuracy)
    i += 1
    if i == 20:
       break
       
print("Validation train top: accuracy = %f  ;  loss = %f" % (np.mean(accuracies_train_top_unfreeze), 
                                                             np.mean(losses_train_top_unfreeze)))


# In[35]:


predictions_train_top_unfreeze = tf_model.predict_classes(X_test, batch_size=32, verbose=0)
for i in range(32):
    show_sample(X_test[i, :, :, :], y_test[i], prediction=predictions_train_top_unfreeze[i, 0])


# 88% accuracy is better, but maybe if I train it for more epochs I can get it to 90%.

# ##### Train again for another 5 epochs

# In[36]:


tensorboard_callback_tt_unfreeze_plus = TensorBoard(log_dir='./logs/train_top_unfreeze_plus/', histogram_freq=0, 
                                               write_graph=True, write_images=False)
checkpoint_callback_tt_unfreeze_plus = ModelCheckpoint(
                                './models/train_top_unfreeze_plus_weights.{epoch:02d}-{val_acc:.2f}.hdf5', 
                                monitor='val_acc', verbose=0, save_best_only=True, 
                                save_weights_only=False, mode='auto', period=1)

nb_epoch_tt_unfreeze_plus = 5

hist_train_top_unfreeze_plus = tf_model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epoch_tt_unfreeze_plus, 
              callbacks=[tensorboard_callback_tt_unfreeze_plus, checkpoint_callback_tt_unfreeze_plus],
              validation_steps=steps_per_epoch_val)
                                                                                                                                   
pandas.DataFrame(hist_train_top_unfreeze_plus.history).to_csv("./history/train_top_unfreeze_plus.csv")


# ##### Evaluate the model and show some predictions 

# In[37]:


accuracies_train_top_unfreeze_plus = np.array([])
losses_train_top_unfreeze_plus = np.array([])

i=0
for X_batch, Y_batch in validation_generator:
    loss, accuracy = tf_model.evaluate(X_batch, Y_batch, verbose=0)
    losses_train_top_unfreeze_plus = np.append(losses_train_top_unfreeze_plus, loss)
    accuracies_train_top_unfreeze_plus = np.append(accuracies_train_top_unfreeze_plus, accuracy)
    i += 1
    if i == 20:
       break
       
print("Validation train top: accuracy = %f  ;  loss = %f" % (np.mean(accuracies_train_top_unfreeze_plus), 
                                                             np.mean(losses_train_top_unfreeze_plus)))


# In[38]:


predictions_train_top_unfreeze_plus = tf_model.predict_classes(X_test, batch_size=32, verbose=0)
for i in range(32):
    show_sample(X_test[i, :, :, :], y_test[i], prediction=predictions_train_top_unfreeze_plus[i, 0])


# 91% accuracy! =) - not bad! 

# ## Object Detection 

# Now moving on to Object Detection, I will use 3 models. 
# 
# 1. Model 1 is a SSD framework on the MobileNet architecture, trained on COCO and fine-tuned on PASCAL VOC (detects 20 classes)
# 2. GoogleLeNet trained on ImageNet (1,000 classes, but no OD)
# 3. SSD framework on VGG16 architecture, trained on ImageNet 2016 (detects 200 classes)
# 

# In[75]:


# Code here is adapted from code found at: https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/

# import the necessary packages
import numpy as np
import argparse
import time
import cv2


def googlenet(image):
    """This creates a classification model by loading in a Googlenet pre-trained model, 
    trained on the ImageNet dataset. It takes an image as input and returns the predicted 
    classes and confidences"""
    
    # cd2821 - modify args parsing from original code - make it into dict instead since not using
        # command line (Carmem's changes)
    args = {}
    args["prototxt"] = '../deep-learning-opencv/bvlc_googlenet.prototxt'
    args["model"] = '../deep-learning-opencv/bvlc_googlenet.caffemodel'
    args["labels"] = '../deep-learning-opencv/synset_words.txt' 
    args['image'] = image

    # load the input image from disk
    image = cv2.imread(args["image"])

    # load the class labels from disk
    rows = open(args["labels"]).read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

    # our CNN requires fixed spatial dimensions for our input image(s)
    # so we need to ensure it is resized to 224x224 pixels while
    # performing mean subtraction (104, 117, 123) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 224, 224)
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("[INFO] classification took {:.5} seconds".format(end - start))

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    idxs = np.argsort(preds[0])[::-1][:5]

    # loop over the top-5 predictions and display them
    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx],
                preds[0][idx] * 100)
            cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

        # display the predicted label + associated probability to the
        # console	
        print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
            classes[idx], preds[0][idx]))

    # cd2821 - use matplotlib to plot the images instead
    # display the output image
    plt.imshow(image)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    return net


# In[64]:


# Code here is adapted from code found at: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

# import the necessary packages
import numpy as np
import argparse
import cv2

def od_mobile_net_ssd(model, image):
    """This function creates an Object Detection model by loading in a MobileNet pre-trained SSD model, 
    trained on the COCO dataset. It takes an image as input and returns the detections"""
    # cd2821 - modify args parsing from original code - make it into dict instead since not using
        # command line (Carmem's changes)

    args = {}
    args['image'] = image
    
    if model ==1:
        args["model"] = '../object-detection-deep-learning/MobileNetSSD_deploy.caffemodel'
        args["prototxt"] = '../object-detection-deep-learning/MobileNetSSD_deploy.prototxt.txt'
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        # (note: normalization is done via the authors of the MobileNet SSD
        # implementation)
        image = cv2.imread(args["image"])
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)


    elif model ==2: # not using this as it doesnt work, can delete
        args["prototxt"] = '../deep-learning-opencv/bvlc_googlenet.prototxt'
        args["model"] = '../deep-learning-opencv/bvlc_googlenet.caffemodel'
        args["labels"] = '../deep-learning-opencv/synset_words.txt' 
        # load the class labels from disk
        rows = open(args["labels"]).read().strip().split("\n")
        CLASSES = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        # load the input image from disk
        image = cv2.imread(args["image"])
        (h, w) = image.shape[:2]
        # our CNN requires fixed spatial dimensions for our input image(s)
        # so we need to ensure it is resized to 224x224 pixels while
        # performing mean subtraction (104, 117, 123) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 224, 224)
        blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

    
    print("____ Image:", args['image'])
    print("____ Classes list:", CLASSES)
    
    args["confidence"] = 0.2

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

 
    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    print("_____ w:", w)
    print("_____ h:", h)
    print("_____ Detections:")
    print(detections)
    print(len(detections))
    print("detections shape: ", detections.shape)
#     print("detections shape[2]: ", detections.shape[2])
            
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        print("i is: ", i)
        print("confidence:", confidence)

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print("box: ")
            print(box)

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 2)

    # cd2821 - use matplotlib to plot the images instead
    # show the output image
    plt.imshow(image)
#     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    return net, detections, w, h


# ## 1. Run Mobile Net SSD on images

# ##### Load the data in by getting the image file names into a list, so can use functions defined above

# In[94]:


train_data_plastic_dir = '../Images/data_for_modeling/train/plastic/'
train_data_noplastic_dir = '../Images/data_for_modeling/train/noplastic/'
validation_data_plastic_dir = '../Images/data_for_modeling/validation/plastic/'
validation_data_noplastic_dir = '../Images/data_for_modeling/validation/noplastic/'


# In[95]:


import glob
# print(glob.glob(train_data_plastic_dir+ '*.*'))
imgs_train_plastic = glob.glob(train_data_plastic_dir+ '*.*')
imgs_train_noplastic = glob.glob(train_data_noplastic_dir+ '*.*')
imgs_validation_plastic = glob.glob(validation_data_plastic_dir+ '*.*')
imgs_validation_noplastic = glob.glob(validation_data_noplastic_dir+ '*.*')


# In[96]:


len(imgs_train_plastic), len(imgs_train_noplastic), len(imgs_validation_plastic), len(imgs_validation_noplastic)


# ##### Run OD with mobilenet on all training plastic images (keyboard interrupt on submission version to avoid ballooning the jupyter notebook size above 100MBs because can't save then
# 

# In[179]:


# Run OD with mobilenet on all training plastic images
for im in imgs_validation_plastic:
    od_mobile_net_ssd(1, im)


# ##### Run OD with mobilenet on all training noplastic images (keyboard interrupt on submission version to avoid ballooning the jupyter notebook size above 100MBs because can't save then

# In[180]:


# Run OD with mobilenet on all training noplastic images
for im in imgs_validation_noplastic:
    od_mobile_net_ssd(1, im)


# ## 2. Run GoogleNet on images

# ##### Run Googlenet on all training plastic images (keyboard interrupt on submission version to avoid ballooning the jupyter notebook size above 100MBs because can't save then

# In[181]:


# Run GoogleNet on all training plastic images
for im in imgs_validation_plastic:
    googlenet(im)


# ##### Run Googlenet on all training noplastic images (keyboard interrupt on submission version to avoid ballooning the jupyter notebook size above 100MBs because can't save then

# In[182]:


# Run GoogleNet on all training noplastic images
for im in imgs_validation_noplastic:
    googlenet(im)


# ## 3. Run VGG16 SSD trained on ImageNet

# ##### Load in the labels for the pre-trained model from the Caffe repo 

# In[195]:


flabels = open('../caffe/data/ILSVRC2016/labelmap_ilsvrc_det.prototxt', 'r') 
contents = flabels.read()


# In[196]:


items = contents.strip().split('}')


# In[197]:


label_dict = {}
for item in items:
#     print(item.strip('\n').split(' '))
    info = item.strip('\n').split(' ')
    label = info[7].split('\n')[0]
    name = info[4].split('\n')[0].replace('"', '')
    display_name = info[10].replace('"', '')
#     print("label: ", label)
#     print("name: ", name)
#     print("display_name: ", display_name)
    label_dict[int(label)] = { 'name': name, 'display_name': display_name}


# ** The keyerror only happens on the last line because it is just a space, so don't worry, since as per the below it picks up all the 201 categories accurately into the dict

# In[198]:


label_dict


# In[199]:


len(label_dict)


# In[208]:


# Code here is adapated from: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

# import the necessary packages
import numpy as np
import argparse
import cv2

def od_vgg16_ssd(image):
    """This function creates an Object Detection model by loading in a VGG16 pre-trained SSD model, 
    trained on the ImageNet dataset. It takes an image as input and returns the detections"""
    # cd2821 - modify args parsing from original code - make it into dict instead since not using
        # command line (Carmem's changes)
    args = {}
    args['image'] = image
    # cd 2821 - Use the VGG16 pre-trained models, trained on ImageNet that I get from the 
        # Caffe repo (https://github.com/weiliu89/caffe/tree/ssd)
    args["model"] = '../caffe_VGGNetSSD_ILSVR2016_models/VGGNet/ILSVRC2016/SSD_300x300/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel'
    args["prototxt"] = '../caffe_VGGNetSSD_ILSVR2016_models/VGGNet/ILSVRC2016/SSD_300x300/deploy.prototxt'
    
    # cd2821 - Modify the colors to look at the new labels
    COLORS = np.random.uniform(0, 255, size=(len(label_dict), 3))
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    print("____ Image:", args['image'])
#     print("____ Classes list:", label_dict)
    
    args["confidence"] = 0.05


    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    print("_____ w:", w)
    print("_____ h:", h)
    print("_____ Detections:")
#     print(detections)
#     print(len(detections))
    print("detections shape: ", detections.shape)
    print("detections shape[2]: ", detections.shape[2])
    
        
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # cd2821 - Modify this to use the new labels
            # display the prediction
            label = "{}: {:.2f}%".format(label_dict[idx]['display_name'], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[idx], 2)

    # cd2821 - use matplotlib to plot the images instead
    # show the output image
    plt.imshow(image)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    return net, detections, w, h


# ##### Run SSD VGG16 on all training plastic images (keyboard interrupt on submission version to avoid ballooning the jupyter notebook size above 100MBs because can't save then

# In[211]:


for im in imgs_train_plastic:
    od_vgg16_ssd(im)

