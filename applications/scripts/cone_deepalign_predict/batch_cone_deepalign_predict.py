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

    numBatch=1
    Nexp = mdExp.size()
    if Nexp<numBatch:
        oneXExp = np.zeros((Nexp,Xdim,Xdim,1),dtype=np.float64)
        YpredAux = np.zeros((Nexp,numClassif),dtype=np.float64) 
    if Nexp>numBatch:
        oneXExp = np.zeros((1,Xdim,Xdim,1),dtype=np.float64)
        YpredAux = np.zeros((numClassif),dtype=np.float64) 
    Ypred = np.zeros((Nexp),dtype=np.float64)    
    refPred = np.zeros((Nexp,3),dtype=np.int64)    
    models=[]

    for i in range(numClassif):
	if os.path.exists(os.path.join(fnODir,'modelCone%d.h5'%(i+1))):
	    models.append(load_model(os.path.join(fnODir,'modelCone%d.h5'%(i+1))))

    idxExp = 0
    for objIdExp in mdExp:
	fnExp = mdExp.getValue(xmippLib.MDL_IMAGE,objIdExp)
	Iexp = xmippLib.Image(fnExp)
        oneXExp[0,:,:,0] = Iexp.getData()
	oneXExp[0,:,:,0] = (oneXExp[0,:,:,0]-np.mean(oneXExp[0,:,:,0]))/np.std(oneXExp[0,:,:,0])
	#if (idxExp%numBatch)==0 or idxExp==(Nexp-1):
        for i in range(numClassif):
	    model = models[i]
            YpredAux[i] = model.predict([oneXExp])
	#print(objIdExp, YpredAux)
        Ypred[idxExp] = np.max(YpredAux)
	refPred[idxExp, :] = [objIdExp, np.argmax(YpredAux)+1, Ypred[idxExp]]
	#print(idxExp, Ypred[idxExp], refPred[idxExp])
        #mdExp.setValue(xmippLib.MDL_MAXCC, float(maxYpred), objIdExp)
        #mdExp.setValue(xmippLib.MDL_REF, int(maxPos)+1, objIdExp) #+1 si o no
	idxExp+=1
    #mdExp.write(fnOName)
    np.savetxt(os.path.join(fnODir,'conePrediction.txt'), refPred)








