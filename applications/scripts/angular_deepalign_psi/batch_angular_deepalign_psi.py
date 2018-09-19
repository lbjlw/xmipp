#!/bin/env python2

#from keras.callbacks import TensorBoard, ModelCheckpoint
#from keras.models import Model
#from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
#import tensorflow as tf
import cv2
import math
import numpy as np
import os
import random
import string
import sys
import xmippLib
import time

batch_size = 128 # Number of boxes per batch

def generateData(fnXmd, maxShift, Nrepeats=10):
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

        for i in range(Nrepeats):
            psi = np.random.uniform(0.0,360.0)*math.pi/180.0
            deltaX = np.random.uniform(-maxShift*Xdim,maxShift*Xdim)
            deltaY = np.random.uniform(-maxShift*Xdim,maxShift*Xdim)

            c = math.cos(psi)
            s = math.sin(psi)
            M = np.float32([[c,s,deltaX],[-s,c,deltaY]])
            X[idx,:,:,0] = cv2.warpAffine(I.getData(),M,(Xdim,Ydim))

            Y[idx,0]=c
            Y[idx,1]=s
            idx+=1
    return X,Y,Xdim

# def constructModel(XDim):
#     inputLayer = Input(shape=(Xdim,Xdim,1), name="input")
#     L = Conv2D(64, (11,11)) (inputLayer)
#     L = BatchNormalization()(L)
#     L = Dropout()(L)
#     L = Conv2D(64, (11,11)) (inputLayer)
#     L = BatchNormalization()(L)
#     L = Dropout()(L)
#     L = Flatten() (L)
#     L = Dense(2,name="output", activation="linear") (L)
#     return Model(inputLayer, L)

if __name__=="__main__":
    fnXmd = sys.argv[1]
    maxShift = float(sys.argv[2])
    X, Y, Xdim = generateData(fnXmd, maxShift)

    # model = constructModel(boxDim)
    # model.summary()
    # model.compile(loss='mean_squared_error', optimizer='Adam')
    # checkpoint = ModelCheckpoint('detectHand_%1.1f.h5'%maxRes, monitor='loss', verbose=1, save_best_only=True)
    # tensorboard = TensorBoard(log_dir="/home/coss/tmp/tb", histogram_freq=1, write_graph=False, write_images=False)
    # model.fit(X, Y, batch_size=128, epochs=10000, verbose=1, validation_split=0.1,
    #                     callbacks=[checkpoint, tensorboard])
    # model.save('detectHand_%1.1f.h5'%maxRes,custom_objects={'coss_loss': coss_loss})
    #model.save('detectHand_%1.1f.h5'%maxRes)
