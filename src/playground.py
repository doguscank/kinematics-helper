from transform import *
from integrator import *
import numpy as np
import math
from math import sin, cos, atan2

def Week2ConceptCheck9(t):
	X = np.float32([sin(0.1 * t), 0.01, cos(0.1 * t)]) * 20

	return X

def Week3ConceptCheck8(t):
	W = np.float32([0, sin(0.1 * t), 0.01, cos(0.1 * t)]) * deg2rad(20)

	return W

def Week3ConceptCheck13(t, q):
	W = np.float32([sin(0.1 * t), 0.01, cos(0.1 * t)]) * deg2rad(3)
	x = np.eye(3) + tilde(q.T) + np.multiply(q, q.T)
	
	dq = 0.5 * np.dot(x, W)

	return dq

DCM = np.float32([[0.763314,-0.568047,-0.307692],
	[0.0946746,-0.372781,0.923077],
	[-0.639053,-0.733728,-0.230769]])

r = DCM2MRP(DCM.T)

print(r)