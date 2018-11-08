#!/usr/bin/env python2

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import Adam
import tensorflow as tf
import cv2
import skimage
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
    fnProj = sys.argv[1]
    fnExp = sys.argv[2]
    fnOName = sys.argv[3]
    fnODir = sys.argv[4]
    Xdim = int(sys.argv[5])

    print('Predict mode')
    newImage = xmippLib.Image()
    copy(fnExp, fnOName)
    mdExp = xmippLib.MetaData(fnOName)
    mdProj = xmippLib.MetaData(fnProj)
    #XProj, XExp, Xdim, Ydim, Nproj, Nexp = loadData(mdProj, mdExp)
    #AJ no quiero cargar todas las imagenes en memoria
    #si puedo cargar todas las projs e iterar por las exp

    Nproj = mdProj.size()
    oneXExp = np.zeros((1,Xdim,Xdim,1),dtype=np.float64)
    oneXProj = np.zeros((1,Xdim,Xdim,1),dtype=np.float64)

    if os.path.exists(os.path.join(fnODir,'modelCorr.h5')):
        model = load_model(os.path.join(fnODir,'modelCorr.h5'))
    
    idxExp = 0
    idxProj = 0
    #allExpFns = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
    #allExpIds = mdExp.getColumnValues(xmippLib.MDL_IDX)
    #allProjFns = mdProj.getColumnValues(xmippLib.MDL_IMAGE)
    for objIdExp in mdExp:
        idxProj = 0
	fnExp = mdExp.getValue(xmippLib.MDL_IMAGE,objIdExp)
	#print(fnExp)
	Iexp = xmippLib.Image(fnExp)
        oneXExp[0,:,:,0] = Iexp.getData()
	#oneXExp[0,:,:,0] = (oneXExp[0,:,:,0]-np.mean(oneXExp[0,:,:,0]))/np.std(oneXExp[0,:,:,0])
	Ypred = np.zeros((Nproj),dtype=np.float64)
	for objId in mdProj:
	    fnProj = mdProj.getValue(xmippLib.MDL_IMAGE,objId)
	    #print(fnProj)
	    Iproj = xmippLib.Image(fnProj)
            oneXProj[0,:,:,0] = Iproj.getData()
            Ypred[idxProj] = model.predict([oneXExp, oneXProj])
	    idxProj+=1
	#print(idxExp, idxProj, Ypred)
        maxYpred = np.max(Ypred)
	maxPos = np.argmax(Ypred)
	print(idxExp, maxYpred, maxPos)
        mdExp.setValue(xmippLib.MDL_MAXCC, float(maxYpred), objIdExp)
        mdExp.setValue(xmippLib.MDL_REF, int(maxPos)+1, objIdExp) #+1 si o no
	idxExp+=1
	if idxExp==1:
	    break
    mdExp.write(fnOName)








