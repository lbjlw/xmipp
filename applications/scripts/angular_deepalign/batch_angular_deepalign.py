#!/usr//bin/env python2

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

batch_size = 128 # Number of boxes per batch

def generateData(fnXmd, maxShift, maxPsi, mode, Nrepeats=30):
    mdIn = xmippLib.MetaData(fnXmd)
    X=None
    Y=None
    Nimgs = mdIn.size()
    idx = 0
    keepAngle = mode=="psi"
    for objId in mdIn:
        fnImg = mdIn.getValue(xmippLib.MDL_IMAGE,objId)
        I = xmippLib.Image(fnImg)
        if X is None:
            Xdim, Ydim, _, _ = I.getDimensions()
            X = np.zeros((Nimgs*Nrepeats,Xdim,Ydim,1),dtype=np.float64)
            Y = np.zeros((Nimgs*Nrepeats,2),dtype=np.float64)

        for i in range(Nrepeats):
            psi = np.random.uniform(-maxPsi,maxPsi)*math.pi/180.0
            deltaX = np.random.uniform(-maxShift,maxShift)
            deltaY = np.random.uniform(-maxShift,maxShift)

            c = math.cos(psi)
            s = math.sin(psi)
            M = np.float32([[c,s,deltaX],[-s,c,deltaY]])
            X[idx,:,:,0] = cv2.warpAffine(I.getData(),M,(Xdim,Ydim))

            if keepAngle:
                Y[idx,0]=c
                Y[idx,1]=s
            else:
                Y[idx,0]=deltaX
                Y[idx,1]=deltaY
            idx+=1
	print(X.shape)
    return X,Y,Xdim


def loadData(fnXmd, maxShift, maxPsi):
    mdIn = xmippLib.MetaData(fnXmd)
    X=None
    Nimgs = mdIn.size()
    idx = 0
    for objId in mdIn:
        fnImg = mdIn.getValue(xmippLib.MDL_IMAGE,objId)
        I = xmippLib.Image(fnImg)
        if X is None:
            Xdim, Ydim, _, _ = I.getDimensions()
            X = np.zeros((Nimgs,Xdim,Ydim,1),dtype=np.float64)
        psi = np.random.uniform(-maxPsi,maxPsi)*math.pi/180.0
        deltaX = np.random.uniform(-maxShift,maxShift)
        deltaY = np.random.uniform(-maxShift,maxShift)

        c = math.cos(psi)
        s = math.sin(psi)
        M = np.float32([[c,s,deltaX],[-s,c,deltaY]])
	if (idx==0):
	    print(M.shape, I.getData().shape, Xdim, Ydim)
        X[idx,:,:,0] = cv2.warpAffine(I.getData(),M,(Xdim,Ydim))
	print(X.shape)
        idx+=1
        print("Ground Truth: c=",c," s=",s," deltaX=",deltaX," deltaY=", deltaY)
    return X, Xdim



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
    train = sys.argv[7]

    if (train=="train"):
        print('Train mode')
        X, Y, Xdim = generateData(fnXmd, maxShift, maxPsi, mode)
        model = constructModel(Xdim)
        model.summary()
        optimizer = Adam(lr=0.0001)
        model.compile(loss='mean_absolute_error', optimizer='Adam')
        history = model.fit(X, Y, batch_size=256, epochs=2, verbose=1, validation_split=0.1) #epochs=15
        myValLoss=np.zeros((1))
        myValLoss[0] = history.history['val_loss'][-1]
        np.savetxt(os.path.join(fnODir,modelFn+'.txt'), myValLoss)
        model.save(os.path.join(fnODir,modelFn+'.h5'))
    else:
        print('Predict mode')
	X, Xdim = loadData(fnXmd, maxShift, maxPsi)
        #model = load_model(os.path.join(fnODir,modelFn+'.h5'))
	model = load_model('/home/ajimenez/ScipionUserData/projects/testDeepAlignment/Runs/000152_XmippProtDeepAlignment3D/extra/psi_iter000000.h5')
        #loss, acc = model.evaluate(X, Y)
        #print('loss=', loss, ' acc=', acc)
        Ypred = model.predict(X)
	print("Ypred psi ", Ypred)
        M = np.float32([[Ypred[0,0],-Ypred[0,1],0],[Ypred[0,1],Ypred[0,0],0]])
	print(M.shape, X.shape, Xdim)
	print(X.shape)
        newX = cv2.warpAffine(X,M,(Xdim,Xdim))

	model = load_model('/home/ajimenez/ScipionUserData/projects/testDeepAlignment/Runs/000152_XmippProtDeepAlignment3D/extra/shift_iter000000.h5')
        #loss, acc = model.evaluate(X, Y)
        #print('loss=', loss, ' acc=', acc)
        Ypred = model.predict(newX)
	print("Ypred shift ", Ypred)
        M = np.float32([[1.0,0.0,Ypred[0,0]],[0.0,1.0,Ypred[0,1]]])
        newX = cv2.warpAffine(newX,M,(Xdim,Xdim))




