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
#import pyworkflow.em.metadata as md
from shutil import copy


def loadData(mdIn, mdExp):
    XProj=None
    XExp=None
    Nproj = mdIn.size()
    Nexp = mdExp.size()
    idx = 0
    for objId in mdIn:
        fnImg = mdIn.getValue(xmippLib.MDL_IMAGE,objId)
        I = xmippLib.Image(fnImg)
        if XProj is None:
            Xdim, Ydim, _, _ = I.getDimensions()
            XProj = np.zeros((Nproj,Xdim,Ydim,1),dtype=np.float64)
	XProj[idx,:,:,0] = I.getData()
        idx+=1

    idx = 0
    for objId in mdExp:
        fnImg = mdExp.getValue(xmippLib.MDL_IMAGE,objId)
        I = xmippLib.Image(fnImg)
        if XExp is None:
            Xdim, Ydim, _, _ = I.getDimensions()
            XExp = np.zeros((Nexp,Xdim,Ydim,1),dtype=np.float64)
	XExp[idx,:,:,0] = I.getData()
        idx+=1

    return XProj, XExp, Xdim, Ydim, Nproj, Nexp


if __name__=="__main__":
    fnExp = sys.argv[1]
    #fnOName = sys.argv[2]
    fnODir = sys.argv[2]
    Xdim = int(sys.argv[3])
    numClassif = int(sys.argv[4])

    print('Predict mode')
    newImage = xmippLib.Image()
    #copy(fnExp, fnOName)
    mdExp = xmippLib.MetaData(fnExp)
    #XProj, XExp, Xdim, Ydim, Nproj, Nexp = loadData(mdProj, mdExp)
    #AJ no quiero cargar todas las imagenes en memoria
    #si puedo cargar todas las projs e iterar por las exp

    sizeBatch=1000
    Nexp = mdExp.size()
    maxBatchs=np.ceil(float(Nexp)/float(sizeBatch))
    Ypred = np.zeros((Nexp),dtype=np.float64)   
    refPred = np.zeros((Nexp,3),dtype=np.float64)    
    models=[]
    for i in range(numClassif):
	if os.path.exists(os.path.join(fnODir,'modelCone%d.h5'%(i+1))):
	    models.append(load_model(os.path.join(fnODir,'modelCone%d.h5'%(i+1))))
    if Nexp>sizeBatch:
        oneXExp = np.zeros((sizeBatch,Xdim,Xdim,1),dtype=np.float64)
        YpredAux = np.zeros((sizeBatch,numClassif),dtype=np.float64)

    idxExp = 0
    countBatch=0
    numBatch = 0
    done = 0
    for objIdExp in mdExp:
	if numBatch==(maxBatchs-1) and done==0:
	    oneXExp = np.zeros((Nexp-idxExp,Xdim,Xdim,1),dtype=np.float64)
            YpredAux = np.zeros((Nexp-idxExp,numClassif),dtype=np.float64)
	    done=1
	fnExp = mdExp.getValue(xmippLib.MDL_IMAGE,objIdExp)
	Iexp = xmippLib.Image(fnExp)
        oneXExp[countBatch,:,:,0] = Iexp.getData()
	oneXExp[countBatch,:,:,0] = (oneXExp[countBatch,:,:,0]-np.mean(oneXExp[countBatch,:,:,0]))/np.std(oneXExp[countBatch,:,:,0])
	countBatch+=1
	idxExp+=1
	refPred[idxExp-1,0] = objIdExp
	if ((idxExp%sizeBatch)==0 or idxExp==Nexp):
	    countBatch = 0
            for i in range(numClassif):
	        model = models[i]
                out = model.predict([oneXExp])
		YpredAux[:,i] = out[:,0]
	    #print("AQUIIII ",idxExp-sizeBatch, idxExp, YpredAux[:,i].shape, out[:,0].shape, Nexp-idxExp+1, numBatch*sizeBatch, Nexp-1)
	    if numBatch==(maxBatchs-1):
                Ypred[numBatch*sizeBatch:Nexp] = np.max(YpredAux, axis=1)
	        refPred[numBatch*sizeBatch:Nexp, 1] = np.argmax(YpredAux, axis=1)+1
	        refPred[numBatch*sizeBatch:Nexp, 2] = Ypred[numBatch*sizeBatch:Nexp]
	    else:
                Ypred[idxExp-sizeBatch:idxExp] = np.max(YpredAux, axis=1)
	        refPred[idxExp-sizeBatch:idxExp, 1] = np.argmax(YpredAux, axis=1)+1
	        refPred[idxExp-sizeBatch:idxExp, 2] = Ypred[idxExp-sizeBatch:idxExp]
	    numBatch+=1

    np.savetxt(os.path.join(fnODir,'conePrediction.txt'), refPred)








