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

batch_size = 128 # Number of boxes per batch



def loadData(mdIn, fnXmd, maxShift, maxPsi):
    X=None
    Nimgs = mdIn.size()
#    newImage = xmippLib.Image()
    idx = 0
    for objId in mdIn:
        fnImg = mdIn.getValue(xmippLib.MDL_IMAGE,objId)
        I = xmippLib.Image(fnImg)
        if X is None:
            Xdim, Ydim, _, _ = I.getDimensions()
#	    Xdim2=Xdim/2
#            Ydim2=Ydim/2
            X = np.zeros((Nimgs,Xdim,Ydim,1),dtype=np.float64)
#	psiDeg = np.random.uniform(-maxPsi,maxPsi)
#        psi = psiDeg*math.pi/180.0
#        deltaX = np.random.uniform(-maxShift,maxShift)
#        deltaY = np.random.uniform(-maxShift,maxShift)

	# Saving data in the MD just for debugging purposes
#	mdIn.setValue(xmippLib.MDL_ANGLE_PSI, float(psiDeg), objId)
#	mdIn.setValue(xmippLib.MDL_SHIFT_X, float(deltaX), objId)
#	mdIn.setValue(xmippLib.MDL_SHIFT_Y, float(deltaY), objId)

#        c = math.cos(psi)
#        s = math.sin(psi)
#        M = np.float32([[c,s,(1-c)*Xdim2-s*Ydim2+deltaX],[-s,c,s*Xdim2+(1-c)*Ydim2+deltaY]])
#        myImage = cv2.warpAffine(I.getData(),M,(Xdim,Ydim)) #+ np.random.normal(0.0, 10.0, [Xdim, Xdim])
#	X[idx,:,:,0] = myImage
#	newImage.setData(myImage)
#	strFnOut = ('%06d@'+fnODir)%(idx+1)
#	newImage.write(os.path.join(strFnOut,'inputImages.stk'))

	X[idx,:,:,0] = I.getData() #+ np.random.normal(0.0, 10.0, [Xdim, Xdim])
        idx+=1
#        print("Ground Truth: psiDeg=", psiDeg," c=",c," s=",s," deltaX=",deltaX," deltaY=", deltaY)
#    mdIn.write(fnXmd)
    return X, Xdim, Ydim, Nimgs


def matrix2str(M):
    mystr = '[['+str(M[0,0])+','+str(M[0,1])+','+str(M[0,2])+']['+str(M[1,0])+','+str(M[1,1])+','+str(M[1,2])+']]'
    return mystr

def str2matrix(mystr):
    mystrCp=''
    for i in range(len(mystr)):
        if mystr[i]=='[' or mystr[i]==']' or mystr[i]==',':
	    mystrCp = mystrCp + ' '
        else:
	    mystrCp = mystrCp + mystr[i]
    mylist = mystrCp.split(' ')	    	
    myListNum = []
    for num in mylist:
        if num!='':
            myListNum.append(float(num))
    if len(myListNum)==9:
    	myListNum = np.asarray(myListNum[:-3])
    elif len(myListNum)==6:
	myListNum = np.asarray(myListNum)
    M = myListNum.reshape((2,3))
    return M


if __name__=="__main__":
    fnXmd = sys.argv[1]
    maxShift = float(sys.argv[2])
    maxPsi = float(sys.argv[3])
    fnODir = sys.argv[4]
    numIter = int(sys.argv[5])

    print('Predict mode')
    newImage = xmippLib.Image()
    mdIn = xmippLib.MetaData(fnXmd)
    X, Xdim, Ydim, Ndim = loadData(mdIn, fnXmd, maxShift, maxPsi)

    Xdim2=Xdim/2
    Ydim2=Ydim/2
    oneXprev = np.zeros((1,Xdim,Ydim,1),dtype=np.float64)
    Xdef = np.zeros((1,Xdim,Ydim,1),dtype=np.float64)
    MIdentity = np.float32([[1.0,0.0,0.0],[0.0,1.0,0.0]])

    flagRot=False
    flagTilt=False
    if os.path.exists(os.path.join(fnODir,'rot_iter%06d.h5'%(numIter))):
        modelRot = load_model(os.path.join(fnODir,'rot_iter%06d.h5'%(numIter)))
	print("ROT ", os.path.join(fnODir,'rot_iter%06d.h5'%(numIter)))
	flagRot=True

    if os.path.exists(os.path.join(fnODir,'tilt_iter%06d.h5'%(numIter))):
        modelTilt = load_model(os.path.join(fnODir,'tilt_iter%06d.h5'%(numIter)))
	print("TILT ", os.path.join(fnODir,'tilt_iter%06d.h5'%(numIter)))
	flagTilt=True

    for i in range(numIter*2):

	if i%2==0:
	    #We are in psi prediction
            #model = load_model('/home/ajimenez/ScipionUserData/projects/testDeepAlignment/Runs/000152_XmippProtDeepAlignment3D/extra/psi_iter%06d.h5'%(i//2))
	    print(os.path.join(fnODir,'psi_iter%06d.h5'%(i//2)))
	    model = load_model(os.path.join(fnODir,'psi_iter%06d.h5'%(i//2)))
	else:
	    #We are in shift prediction
            #model = load_model('/home/ajimenez/ScipionUserData/projects/testDeepAlignment/Runs/000152_XmippProtDeepAlignment3D/extra/shift_iter%06d.h5'%(i//2))
	    print(os.path.join(fnODir,'shift_iter%06d.h5'%(i//2)))
	    model = load_model(os.path.join(fnODir,'shift_iter%06d.h5'%(i//2)))
	
	idx=0
	for objId in mdIn:
	    if i==0:
		mdIn.setValue(xmippLib.MDL_TRANSFORM_MATRIX, matrix2str(MIdentity), objId) #MDL_TRANSFORM_MATRIX with matrix for MD
		mdIn.setValue(xmippLib.MDL_COMMENT, matrix2str(MIdentity), objId) #MDL_COMMENT with matrix for centered transformations with warpAffine
	    oneX = X[idx,:,:,0]
	    Mstr = mdIn.getValue(xmippLib.MDL_COMMENT, objId)
	    #print(i,idx,Mstr)
	    Mprev = str2matrix(Mstr)
            oneXprev[0,:,:,0] = cv2.warpAffine(oneX,Mprev,(Xdim,Ydim))
	    Ypred = model.predict(oneXprev)
	    #print(i,idx,Ypred)
	    if i%2==0:
	        #We are in psi prediction
	        c = Ypred[0,0]
	        s = -Ypred[0,1]
		psi = np.arctan2(s,c)
		c = math.cos(psi)
		s = math.sin(psi)
	        Mnew = np.float32([[c,s,(1-c)*Xdim2-s*Ydim2],[-s,c,s*Xdim2+(1-c)*Ydim2]])
		Mnew_MD = np.float32([[c,s,0.0],[-s,c,0.0]])
	    else:
	        #We are in shift prediction
	        deltaX = -Ypred[0,0]
	        deltaY = -Ypred[0,1]
	        Mnew = np.float32([[1.0,0.0,deltaX],[0.0,1.0,deltaY]])
		Mnew_MD = Mnew #np.float32([[1.0,0.0,Ypred[0,0]],[0.0,1.0,Ypred[0,1]]])
	    #Saving matrix in MDL_COMMENT for centered transformations with warpAffine
	    M = np.dot(np.vstack((Mnew,[0.0, 0.0, 1.0])), np.vstack((Mprev,[0.0, 0.0, 1.0])))
	    mdIn.setValue(xmippLib.MDL_COMMENT, matrix2str(M), objId)
	    #Saving matrix in MDL_TRANSFORM_MATRIX for MD
	    Mstr_MD = mdIn.getValue(xmippLib.MDL_TRANSFORM_MATRIX, objId)
	    Mprev_MD = str2matrix(Mstr_MD)
	    M_MD = np.dot(np.vstack((Mnew_MD,[0.0, 0.0, 1.0])), np.vstack((Mprev_MD,[0.0, 0.0, 1.0])))
	    mdIn.setValue(xmippLib.MDL_TRANSFORM_MATRIX, matrix2str(M_MD), objId)

	    if i==((numIter*2)-1):
		Mstr = mdIn.getValue(xmippLib.MDL_COMMENT, objId)
	        Mdef = str2matrix(Mstr)
                Xdef[0,:,:,0] = cv2.warpAffine(oneX,Mdef,(Xdim,Ydim))
	        newImage.setData(Xdef[0,:,:,0])
	        strFnOut = ('%06d@'+fnODir)%(idx+1)
	        newImage.write(os.path.join(strFnOut,'outputImages.stk'))
		#mdIn.setValue(xmippLib.MDL_IMAGE, os.path.join(strFnOut,'outputImages.stk'), objId)

		Mstr2 = mdIn.getValue(xmippLib.MDL_TRANSFORM_MATRIX, objId)
	        Mdef2 = str2matrix(Mstr2)
		psi = np.rad2deg(np.arctan2(Mdef2[0,1],Mdef2[0,0]))
		mdIn.setValue(xmippLib.MDL_ANGLE_PSI, float(-psi), objId)
		mdIn.setValue(xmippLib.MDL_SHIFT_X, float(Mdef2[0,2]), objId)
		mdIn.setValue(xmippLib.MDL_SHIFT_Y, float(Mdef2[1,2]), objId)	
    		
		#Checking if we have the models for 3D reconstruction
		if flagRot:
		    YpredRot = modelRot.predict(Xdef)
            	    c = YpredRot[0,0]
            	    s = YpredRot[0,1]
	    	    rot = np.rad2deg(np.arctan2(s,c))
	    	    mdIn.setValue(xmippLib.MDL_ANGLE_ROT, float(rot), objId)
		if flagTilt:
		    YpredTilt = modelTilt.predict(Xdef)
            	    c = YpredTilt[0,0]
            	    s = YpredTilt[0,1]
	    	    tilt = np.rad2deg(np.arctan2(s,c))
	    	    mdIn.setValue(xmippLib.MDL_ANGLE_TILT, float(tilt), objId)

	    idx+=1
	mdIn.write(fnXmd)








