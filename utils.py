# -*- coding: utf-8 -*-
# @Author: ahpalmerUNR
# @Date:   2020-06-04 11:07:18
# @Last Modified by:   ahpalmerUNR
# @Last Modified time: 2020-07-20 18:02:15
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

def calculateNormBetweenSets(set1,set2):
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
	first = np.ones((1,7))
	second = np.array([a for a in range(7)])
	assert(((first - second)**2).sum()**(0.5) == calculateNormBetweenSets(first,second))
	transformationMatrix = np.matrix([[  0.8365163, -0.1950597,  0.5120471, .3],[0.4829629,  0.7038788, -0.5208661, 1.2],[-0.2588190,  0.6830127,  0.6830127, 5 ],[0,0,0,1]])
	transformationMatrixInverse = np.linalg.inv(transformationMatrix)
	horizonPoint1 = np.array([[.1,0.0,1,1]])
	horizonPoint2 = np.array([[-.2,0.0,1,1]])
	horizonPoint3 = np.array([[-.2,0.0,1.2,1]])
	print(horizonPoint1,horizonPoint2,horizonPoint3)
	transHPoint1 = applyTransformationToPoints(transformationMatrix,horizonPoint1)[:3]
	transHPoint2 = applyTransformationToPoints(transformationMatrix,horizonPoint2)[:3]
	transHPoint3 = applyTransformationToPoints(transformationMatrix,horizonPoint3)[:3]
	print(transHPoint1,transHPoint2,transHPoint3)
	print(transHPoint1 - transHPoint2)
	print(transHPoint1 - transHPoint3)
	print(transHPoint2 - transHPoint3)
	print(np.cross((transHPoint1 - transHPoint3).T,(transHPoint1 - transHPoint2).T))#for normal to plane

if __name__ == '__main__':
	unitTest()
