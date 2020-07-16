# -*- coding: utf-8 -*-
# @Author: ahpalmerUNR
# @Date:   2020-06-04 11:07:18
# @Last Modified by:   ahpalmer
# @Last Modified time: 2020-07-16 15:35:30
import numpy as np
import math
import random as rm

def projectPointsOntoPlaneUsingMatrix(pointsToProject,transformationMatrix):
	pointSlopes = getTransformedSlopes(pointsToProject,transformationMatrix)
	projectedPoints =projectSlopesOntoPlane(pointSlopes,transformationMatrix[:,3])
	return projectedPoints

def getTransformedSlopes(pointsToProject,transformationMatrix):
	translationXYZ = transformationMatrix[:,3]
	pointsWithOnesRow = np.concatenate((pointsToProject,np.ones((pointsToProject.shape[0],1))),axis=1)
	transformedPoints = applyTransformationToPoints(transformationMatrix,pointsWithOnesRow)
	return getPointSlopesFromCameraCenter(transformedPoints,translationXYZ)

def applyTransformationToPoints(transformationMatrix,pointsToTransform):
	return np.dot(transformationMatrix,pointsToTransform.T)

def getPointSlopesFromCameraCenter(points,cameraTranslation):
	return (points - np.reshape(cameraTranslation,(1,4)).T).T

def projectSlopesOntoPlane(pointSlopes,cameraTranslation):
	projectionsXs = np.asarray(cameraTranslation[0] - (pointSlopes[:,0]/pointSlopes[:,2])*cameraTranslation[2]).T.squeeze()
	projectionsYs = np.asarray(cameraTranslation[1] - (pointSlopes[:,1]/pointSlopes[:,2])*cameraTranslation[2]).T.squeeze()
	projections = np.array([projectionsXs,projectionsYs,np.zeros(projectionsXs.shape[0]),np.ones(projectionsXs.shape[0])])
	return projections.T.squeeze()

def calculateNormBetweenSets3DPoints(set1,set2):
	return np.linalg.norm(set1-set2)

def addColumnOfOnesToNumpy2DArray(arrayIn):
	rows = arrayIn.shape[0]
	return np.concatenate((arrayIn,np.ones((rows,1))),axis=1)

def getDetectablePointCloudFromLaser(laserScan):
	cloud = []
	for index in range(len(laserScan.ranges)):
		if laserScan.ranges[index]>=laserScan.range_max or laserScan.ranges[index]<=laserScan.range_min:
			continue
		else:
			cloud.append((laserScan.ranges[index]*math.cos(laserScan.angle_min + index*laserScan.angle_increment),laserScan.ranges[index]*math.sin(laserScan.angle_min + index*laserScan.angle_increment),0))

	return np.array(cloud)

def getPointCloudFromLaser(laserScan):
	cloud = []
	for index in range(len(laserScan.ranges)):
		if laserScan.ranges[index]>=laserScan.range_max or laserScan.ranges[index]<=laserScan.range_min:
			cloud.append((laserScan.range_max*math.cos(laserScan.angle_min + index*laserScan.angle_increment),laserScan.range_max*math.sin(laserScan.angle_min + index*laserScan.angle_increment),0))
		else:
			cloud.append((laserScan.ranges[index]*math.cos(laserScan.angle_min + index*laserScan.angle_increment),laserScan.ranges[index]*math.sin(laserScan.angle_min + index*laserScan.angle_increment),0))

	return np.array(cloud)

def unitTest():
	transformationMatrix = np.eye(4)
	transformationMatrix[2,3] = 1
	pointsToTransform = np.random.rand(5,3)
	pointsToTransform[:,2] = 1
	projectedPoints = projectPointsOntoPlaneUsingMatrix(pointsToTransform,transformationMatrix)
	print("Identity")
	print(pointsToTransform)
	print(transformationMatrix)
	print(projectedPoints)
	transformationMatrix = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,1],[0,0,0,1]])
	pointsToTransform = np.random.rand(5,3)
	pointsToTransform[:,0] = 1
	projectedPoints = projectPointsOntoPlaneUsingMatrix(pointsToTransform,transformationMatrix)
	print("XYZ==>YZX")
	print(pointsToTransform)
	print(transformationMatrix)
	print(projectedPoints)

if __name__ == '__main__':
	unitTest()
