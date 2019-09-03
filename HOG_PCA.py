# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:59:48 2019

@author: hasham
"""

from __future__ import division
import numpy as np
from cv2 import HOGDescriptor
from sklearn.decomposition import PCA





class DimensionalityReduction:
    
    def __init__(self, winSize, blockSize, blockStride,  cellSize, nbins, imageNormalization, pcaComponent):
        
        self.winsize = winSize
        self.blocksize= blockSize
        self.blockstride= blockStride
        self.cellsize=cellSize
        self.nbin=nbins
        self.imgNormalize=imageNormalization
        self.component=pcaComponent
    
    def ComputeHOG(self, trainData, testData):
        print('ComputeHog Function is called to calculate Histogram of Oriented Gradient')
        trainSize=trainData.shape[0]
        testSize=testData.shape[0]
        if self.imgNormalize is True:
            trainData=trainData*255
            testData=testData*255
            trainData=trainData.astype('uint8')
            testData=testData.astype('uint8')
        else:
            print('Images already Normalized')
        hog = HOGDescriptor(_winSize=(self.winsize,self.winsize), _blockSize=(self.blocksize,self.blocksize),_blockStride=(self.blockstride,self.blockstride), _cellSize=(self.cellsize,self.cellsize), _nbins=self.nbin)
        hogTrain = np.zeros((trainSize, hog.getDescriptorSize()),dtype="float32")
        for i in range(0,trainSize):
            trainImage = trainData[i].reshape(self.winsize,self.winsize)
            h = hog.compute(trainImage)
            hogTrain[i,:] = h[:,0]
        hogTest = np.zeros((testSize, hog.getDescriptorSize()), dtype="float32")
        for i in range(0,testSize):
            testImage = testData[i].reshape(self.winsize,self.winsize)
            h = hog.compute(testImage)
            hogTest[i,:] = h[:,0]
            print ("Extracted HoG features (", h.size, " per image)")
            return hogTrain, hogTest
        
    def ComputePCA(self, trainData, testData):
        print ('ComputePCA Function is called to calculate principal component Analysis')
        pca =PCA(n_components=self.component)
        #min_max_scaler = preprocessing.MinMaxScaler()
        pca.fit(trainData)
        pcaTrain = pca.fit_transform(trainData)
        pcaTest=pca.transform(testData)
        print("original shape of data:   ", trainData.shape)
        print("transformed shape of data:", pcaTrain.shape)
        return pcaTrain, pcaTest






     