#!/usr/bin/env python2

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, concatenate, Subtract
from keras.optimizers import Adam
import tensorflow as tf
import cv2
import math
import numpy as np
import os
import random
import string
import sys
import xmippLib
import time
from keras.models import load_model
from time import time
import keras

batch_size = 1000 # Number of boxes per batch


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, 
                 shuffle, pathsExp, pathsProj):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
	self.pathsExp = pathsExp
	self.pathsProj = pathsProj
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [Xexp, Xproj], y = self.__data_generation(list_IDs_temp)

        return [Xexp, Xproj], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
	Xexp = np.zeros((self.batch_size,self.dim,self.dim,1),dtype=np.float64)
	Xproj = np.zeros((self.batch_size,self.dim,self.dim,1),dtype=np.float64)
        y = np.empty((self.batch_size), dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            Iexp = np.reshape(xmippLib.Image(self.pathsExp[ID]).getData(),(self.dim,self.dim,1))
	    Iexp = (Iexp-np.mean(Iexp))/np.std(Iexp)
            Iproj = np.reshape(xmippLib.Image(self.pathsProj[ID]).getData(),(self.dim,self.dim,1))
	    Iproj = (Iproj-np.mean(Iproj))/np.std(Iproj)
	    Xexp[i,] = Iexp
	    Xproj[i,] = Iproj

            # Store class
            y[i] = self.labels[ID]

        return [Xexp, Xproj], y


def createValidationData(pathsExp, pathsProj, cc_vector, percent=0.1):
    sizeValData = int(round(len(pathsProj)*percent))
    print("sizeValData",sizeValData)
    val_img_proj = []
    val_img_exp = []
    val_labels = []
    for i in range(sizeValData):
	print(i)
	k = 0 #np.random.randint(0,len(pathsProj))
	Iproj = xmippLib.Image(pathsProj[k])
	Iproj = np.reshape(Iproj.getData(),(Xdim,Xdim,1))
	Iproj = (Iproj-np.mean(Iproj))/np.std(Iproj)
	Iexp = xmippLib.Image(pathsExp[k])
	Iexp = np.reshape(Iexp.getData(),(Xdim,Xdim,1))
	Iexp = (Iexp-np.mean(Iexp))/np.std(Iexp)
        val_img_proj.append(Iproj)
        val_img_exp.append(Iexp)
        val_labels.append(cc_vector[k])
	del pathsExp[k]
	del pathsProj[k]
	del cc_vector[k]
    return [np.asarray(val_img_exp).astype('float64'), np.asarray(val_img_proj).astype('float64')], np.asarray(val_labels)
 

def data_gen(pathsProj, pathsExp, cc, Xdim):
    while True:
        batch_img_proj = []
        batch_img_exp = []
        labels = []
        for i in range(batch_size):
            k = np.random.randint(0,len(pathsProj))
	    Iproj = xmippLib.Image(pathsProj[k])
	    Iproj = np.reshape(Iproj.getData(),(Xdim,Xdim,1))
	    Iexp = xmippLib.Image(pathsExp[k])
	    Iexp = np.reshape(Iexp.getData(),(Xdim,Xdim,1))
            batch_img_proj.append(Iproj)
            batch_img_exp.append(Iexp)
            labels.append(cc[k])

        yield [np.asarray(batch_img_exp).astype('float64'), np.asarray(batch_img_proj).astype('float64')], np.asarray(labels)



def generateData(fnXmd, fnXmdExp, fncc):
    mdIn = xmippLib.MetaData(fnXmd)
    mdExp = xmippLib.MetaData(fnXmdExp)
    X1=None
    X2=None
    Y=None
    Nin = mdIn.size()
    Nexp = mdExp.size()
    cc = np.loadtxt(fncc)

    idx = 0
    cont=0
    contExp=0
    for objIdExp in mdExp:
        cont=0
        fnExp = mdExp.getValue(xmippLib.MDL_IMAGE,objIdExp)
        Iexp = xmippLib.Image(fnExp)

	for objId in mdIn:
            fn = mdIn.getValue(xmippLib.MDL_IMAGE,objId)
            I = xmippLib.Image(fn)
            if X1 is None:
                Xdim, Ydim, _, _ = Iexp.getDimensions()
                X1 = np.zeros((Nin*Nexp,Xdim,Ydim,1),dtype=np.float64)
                X2 = np.zeros((Nin*Nexp,Xdim,Ydim,1),dtype=np.float64)
                Y = np.zeros((Nin*Nexp,1),dtype=np.float64)

            X1[idx,:,:,0] = Iexp.getData()
	    X2[idx,:,:,0] = I.getData()
	    #print(contExp, cont)
            Y[idx,0]=cc[cont, contExp]
            idx+=1
	    cont+=1
	contExp+=1


    return X1,X2,Y,Xdim

def constructModel(Xdim):
    inputLayer = Input(shape=(Xdim,Xdim,1), name="input")
    L = Conv2D(16, (32,32), activation="relu", padding="same") (inputLayer)
    L = BatchNormalization()(L)
    L = MaxPooling2D()(L)
    L = Conv2D(32, (16,16), activation="relu", padding="same") (L)
    L = BatchNormalization()(L)
    L = MaxPooling2D()(L)
    L = Dropout(0.2)(L)

    inputLayer2 = Input(shape=(Xdim,Xdim,1), name="input2")
    L2 = Conv2D(16, (32,32), activation="relu", padding="same") (inputLayer2)
    L2 = BatchNormalization()(L2)
    L2 = MaxPooling2D()(L2)
    L2 = Conv2D(32, (16,16), activation="relu", padding="same") (L2)
    L2 = BatchNormalization()(L2)
    L2 = MaxPooling2D()(L2)
    L2 = Dropout(0.2)(L2)
    
    #concatenated = concatenate([L, L2])    
    concatenated = Subtract()([L, L2])    #AJ probar la capa substract con otra convolucion (maxpooling antes?)

    concatenated = Conv2D(32, (8,8), activation="relu", padding="same") (concatenated)
    concatenated = BatchNormalization()(concatenated)
    concatenated = MaxPooling2D()(concatenated)
    concatenated = Conv2D(32, (4,4), activation="relu", padding="same") (concatenated)
    concatenated = BatchNormalization()(concatenated)

    Lout = Flatten() (concatenated)
    Lout = Dense(512, activation="linear")(Lout)
    Lout = BatchNormalization()(Lout)
    Lout = Dropout(0.2)(Lout)
    Lout = Dense(1,name="output", activation="linear")(Lout)
    return Model([inputLayer, inputLayer2], Lout)

if __name__=="__main__":
    fnXmd = sys.argv[1]
    fnXmdExp = sys.argv[2]
    fncc = sys.argv[3]
    fnODir = sys.argv[4]
    modelFn = sys.argv[5]
    numEpochs = int(sys.argv[6])
    Xdim = int(sys.argv[7])

    mdProj = xmippLib.MetaData(fnXmd)
    mdExp = xmippLib.MetaData(fnXmdExp)
    NProj = mdProj.size()
    Nexp = mdExp.size()
    cc = np.loadtxt(fncc)

    #AJ new code to generate data for data_gen
    pathsProj = []
    pathsExp = []
    cc_vector = []
    idx = 0
    cont=0
    contExp=0
    allExpFns = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
    allProjFns = mdProj.getColumnValues(xmippLib.MDL_IMAGE)
    for fnExp in allExpFns:
        cont=0
        #fnExp = mdExp.getValue(xmippLib.MDL_IMAGE,objIdExp)
        #print(fnExp)
	for fnProj in allProjFns:
	    #fnProj = mdProj.getValue(xmippLib.MDL_IMAGE,objId)
            #print(fnProj)
	    pathsExp.append(fnExp)
	    pathsProj.append(fnProj)
            cc_vector.append(cc[cont, contExp])
            idx+=1
	    cont+=1
	contExp+=1
    x_val, y_val = createValidationData(pathsExp, pathsProj, cc_vector, 0.005)
    np.savetxt(os.path.join(fnODir,'pruebaYval.txt'), y_val)
    #END AJ


    # Parameters
    params = {'dim': Xdim,
          'batch_size': 256,
          'shuffle': True,
	  'pathsExp': pathsExp,
	  'pathsProj': pathsProj}
    # Datasets
    partition = range(len(cc_vector))
    labels = cc_vector
    # Generators
    training_generator = DataGenerator(partition, labels, **params)

    print('Train mode')
    #start_time = time()
    #X1, X2, Y, Xdim = generateData(fnXmd, fnXmdExp, fncc)
    #elapsed_time = time() - start_time
    #print("Time in generateData: %0.10f seconds." % elapsed_time)
    start_time = time()
    model = constructModel(Xdim)
    model.summary()
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    #history = model.fit([X1, X2], Y, batch_size=256, epochs=numEpochs, verbose=1, validation_split=0.1)
    #history = model.fit_generator(data_gen(pathsProj, pathsExp, cc_vector, Xdim), batch_size=256, epochs=numEpochs, verbose=1, validation_split=0.1)
    #history = model.fit_generator(data_gen(pathsProj, pathsExp, cc_vector, Xdim), steps_per_epoch=1000, epochs=20, verbose=1, validation_data = (x_val, y_val))

    history = model.fit_generator(generator = training_generator, epochs=10, verbose=1, validation_data = (x_val, y_val))    
    myValLoss=np.zeros((1))
    myValLoss[0] = history.history['val_loss'][-1]
    np.savetxt(os.path.join(fnODir,modelFn+'.txt'), myValLoss)
    model.save(os.path.join(fnODir,modelFn+'.h5'))
    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)

    Ypred = model.predict(x_val)
    np.savetxt(os.path.join(fnODir,'pruebaYpred2.txt'), Ypred)







