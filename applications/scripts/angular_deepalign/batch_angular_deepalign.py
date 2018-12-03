#!/usr/bin/env python2

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
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

batch_size = 128 # Number of boxes per batch


def generateData(fnXmd, maxShift, maxPsi, mode, Nrepeats=30):
    mdIn = xmippLib.MetaData(fnXmd)
    X=None
    Y=None
    Nimgs = mdIn.size()
    idx = 0

    for objId in mdIn:
        fnImg = mdIn.getValue(xmippLib.MDL_IMAGE,objId)
        I = xmippLib.Image(fnImg)
        if X is None:
            Xdim, Ydim, _, _ = I.getDimensions()
            X = np.zeros((Nimgs*Nrepeats,Xdim,Ydim,1),dtype=np.float64)
            Y = np.zeros((Nimgs*Nrepeats,2),dtype=np.float64)
            Xdim2=Xdim/2
            Ydim2=Ydim/2

        for i in range(Nrepeats):
            psi = np.random.uniform(-maxPsi,maxPsi)*math.pi/180.0
            deltaX = np.random.uniform(-maxShift,maxShift)
            deltaY = np.random.uniform(-maxShift,maxShift)
            c = math.cos(psi)
            s = math.sin(psi)
            M = np.float32([[c,s,(1-c)*Xdim2-s*Ydim2+deltaX],[-s,c,s*Xdim2+(1-c)*Ydim2+deltaY]])
            X[idx,:,:,0] = cv2.warpAffine(I.getData(),M,(Xdim,Ydim)) + np.random.normal(0.0, 2.0, [Xdim, Xdim])
	    #A = xmippLib.Image()
	    #A.setData(X[idx,:,:,0])
	    #A.write('./aqui.stk')
	    #aaaaaaaa

            if mode=="psi":
                Y[idx,0]=c
                Y[idx,1]=s
            elif mode=="shift":
                Y[idx,0]=deltaX
                Y[idx,1]=deltaY
            elif mode=="rot":
		rot = mdIn.getValue(xmippLib.MDL_ANGLE_ROT,objId)
		rot = np.deg2rad(rot)
                Y[idx,0] = math.cos(rot)
                Y[idx,1] = math.sin(rot)
            elif mode=="tilt":
		tilt = mdIn.getValue(xmippLib.MDL_ANGLE_TILT,objId)
		tilt = np.deg2rad(tilt)
                Y[idx,0] = math.cos(tilt)
                Y[idx,1] = math.sin(tilt)
            idx+=1
    return X,Y,Xdim

def constructModel(Xdim):
    inputLayer = Input(shape=(Xdim,Xdim,1), name="input")
    L = Conv2D(16, (11,11), activation="relu") (inputLayer)
    L = BatchNormalization()(L)
    L = MaxPooling2D()(L)
    L = Conv2D(16, (5,5), activation="relu") (L)
    L = BatchNormalization()(L)
    L = MaxPooling2D()(L)
    L = Dropout(0.2)(L)
    L = Flatten() (L)
    L = Dense(2,name="output", activation="linear") (L)
    return Model(inputLayer, L)

if __name__=="__main__":
    fnXmd = sys.argv[1]
    maxShift = float(sys.argv[2])
    maxPsi = float(sys.argv[3])
    mode = sys.argv[4]
    fnODir = sys.argv[5]
    modelFn = sys.argv[6]
    numEpochs = int(sys.argv[7])

    print('Train mode')
    start_time = time()
    X, Y, Xdim = generateData(fnXmd, maxShift, maxPsi, mode)
    elapsed_time = time() - start_time
    print("Time in generateData: %0.10f seconds." % elapsed_time)
    start_time = time()
    model = constructModel(Xdim)
    model.summary()
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    history = model.fit(X, Y, batch_size=256, epochs=numEpochs, verbose=1, validation_split=0.1)
    myValLoss=np.zeros((1))
    myValLoss[0] = history.history['val_loss'][-1]
    np.savetxt(os.path.join(fnODir,modelFn+'.txt'), myValLoss)
    model.save(os.path.join(fnODir,modelFn+'.h5'))
    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)






