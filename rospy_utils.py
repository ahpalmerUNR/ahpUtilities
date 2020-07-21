# -*- coding: utf-8 -*-
# @Author: ahpalmerUNR
# @Date:   2020-07-20 13:45:50
# @Last Modified by:   ahpalmerUNR
# @Last Modified time: 2020-07-20 15:33:33
import rospy

def getTimeDifference(priorTime,currentTime):
	if priorTime == rospy.Time(0):
		return priorTime
	return currentTime - priorTime
