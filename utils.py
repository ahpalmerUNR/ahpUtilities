# -*- coding: utf-8 -*-
# @Author: ahpalmerUNR
# @Date:   2020-06-04 11:07:18
# @Last Modified by:   ahpalmerUNR
# @Last Modified time: 2020-07-29 23:09:20
import numpy as np
import math
import random as rm

def getConvertedPixelToNonOpticalCamPoint(pixelPoints,cameraMatrix):
	pixelPointsWithZColumnOnes = addColumnOfOnesToNumpy2DArray(pixelPoints)
	projectingInverseCamMatrix = np.matrix([[0,0,1],[-1,0,0],[0,-1,0]])*np.linalg.inv(cameraMatrix)
	projOfCamPointsInNonOpticalFrame = projectingInverseCamMatrix*pixelPointsWithZColumnOnes.T
	return projOfCamPointsInNonOpticalFrame

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

def getHorizonLine(cameraMatrix,transformationMatrix):
	parallelSets = getParallelSets()
	transformedPoints = getTransformedParallelSets(parallelSets,transformationMatrix)
	cameraPoints = getCameraPoints(cameraMatrix,transformedPoints)
	print(cameraPoints)
	parallelLines = getParallelLineSets(cameraPoints)
	horizonPoints = getHorizonPointsFromParallelLines(parallelLines)
	slopeHorizon,interceptHorizon = getSlopeAndInterceptOf2Points2D(horizonPoints[0],horizonPoints[1])
	return slopeHorizon,interceptHorizon,cameraPoints,parallelLines,horizonPoints

def getParallelSets():
	line1 = np.array([(-.1,0,-.5,1.0),(-.1,0,.6,1)])
	line2 = np.array([(.1,0,-.5,1.0),(.1,0,.6,1)])
	line3 = np.array([(-.1,0,-.5,1.0),(0,0,.6,1)])
	line4 = np.array([(.1,0,-.5,1.0),(.2,0,.6,1)])
	set1 = [line1,line2]
	set2 = [line3,line4]
	return set1,set2

def getTransformedParallelSets(parallelSets,transformationMatrix):
	transformedLine1 = applyTransformationToPoints(transformationMatrix,parallelSets[0][0]).T.A
	transformedLine2 = applyTransformationToPoints(transformationMatrix,parallelSets[0][1]).T.A
	transformedLine3 = applyTransformationToPoints(transformationMatrix,parallelSets[1][0]).T.A
	transformedLine4 = applyTransformationToPoints(transformationMatrix,parallelSets[1][1]).T.A
	transformedSet1 = [transformedLine1[:,:3],transformedLine2[:,:3]]
	transformedSet2 = [transformedLine3[:,:3],transformedLine4[:,:3]]
	return transformedSet1,transformedSet2

def getCameraPoints(cameraMatrix,transformedPoints):
	cameraLine1 = applyTransformationToPoints(cameraMatrix,transformedPoints[0][0]).T.A
	cameraLine1 = (cameraLine1.T/(cameraLine1[:,2])).T
	cameraLine2 = applyTransformationToPoints(cameraMatrix,transformedPoints[0][1]).T.A
	cameraLine2 = (cameraLine2.T/(cameraLine2[:,2])).T
	cameraLine3 = applyTransformationToPoints(cameraMatrix,transformedPoints[1][0]).T.A
	cameraLine3 = (cameraLine3.T/(cameraLine3[:,2])).T
	cameraLine4 = applyTransformationToPoints(cameraMatrix,transformedPoints[1][1]).T.A
	cameraLine4 = (cameraLine4.T/(cameraLine4[:,2])).T
	cameraSet1 = [cameraLine1[:,:2],cameraLine2[:,:2]]
	cameraSet2 = [cameraLine3[:,:2],cameraLine4[:,:2]]
	return cameraSet1,cameraSet2

def getParallelLineSets(cameraPoints):
	parallelLine1 = getSlopeAndInterceptOf2Points2D(cameraPoints[0][0][0],cameraPoints[0][0][1])
	parallelLine2 = getSlopeAndInterceptOf2Points2D(cameraPoints[0][1][0],cameraPoints[0][1][1])
	parallelLine3 = getSlopeAndInterceptOf2Points2D(cameraPoints[1][0][0],cameraPoints[1][0][1])
	parallelLine4 = getSlopeAndInterceptOf2Points2D(cameraPoints[1][1][0],cameraPoints[1][1][1])
	parallelSet1 = [parallelLine1,parallelLine2]
	parallelSet2 = [parallelLine3,parallelLine4]
	return parallelSet1,parallelSet2

def getHorizonPointsFromParallelLines(parallelLines):
	xIntersection1 = getXfromMBOf2Lines(parallelLines[0][0],parallelLines[0][1])
	xIntersection2 = getXfromMBOf2Lines(parallelLines[1][0],parallelLines[1][1])
	yIntersection1 = getYfromXandMB(xIntersection1,parallelLines[0][0])
	yIntersection2 = getYfromXandMB(xIntersection2,parallelLines[1][0])
	horizonPoints = [(xIntersection1,yIntersection1),(xIntersection2,yIntersection2)]
	return horizonPoints


def getSlopeAndInterceptOf2Points2D(point1,point2):
	points = [point1,point2]
	x_cords,y_cords = zip(*points)
	A = np.vstack([x_cords,np.ones(len(x_cords))]).T
	slope, intercept = np.linalg.lstsq(A,y_cords,rcond=None)[0]
	return slope, intercept

def getXfromMBOf2Lines(mbset1,mbset2):
	return (mbset1[1] - mbset2[1])/(mbset2[0] - mbset1[0])

def getYfromXandMB(x,mbset):
	return mbset[0]*x + mbset[1]

def unitTest():
	first = np.ones((1,7))
	second = np.array([a for a in range(7)])
	assert(((first - second)**2).sum()**(0.5) == calculateNormBetweenSets(first,second))
	transformationMatrix = np.matrix([[  0.8365163, -0.1950597,  0.5120471, .3],[0.4829629,  0.7038788, -0.5208661, 1.2],[-0.2588190,  0.6830127,  0.6830127, 5 ],[0,0,0,1]])
	cameraMatrix = np.matrix([[489.333, 0.000000, 320, 0.000000, 367, 180, 0.000000, 0.000000, 1.000000]]).reshape((3,3))
	transformationMatrixInverse = np.linalg.inv(transformationMatrix)
	horizon = getHorizonLine(cameraMatrix,transformationMatrixInverse)
	print(horizon)
	# print(np.cross((transHPoint1 - transHPoint3).T,(transHPoint1 - transHPoint2).T))#for normal to plane

if __name__ == '__main__':
	unitTest()
