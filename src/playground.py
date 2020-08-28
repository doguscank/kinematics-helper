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

def Week3ConceptCheck20(t):
	W = np.float32([sin(0.1 * t), 0.01, cos(0.1 * t)]) * deg2rad(20)

	return W

v1_B = np.float32([[0.8273,0.5541,-0.0920]])
v1_N = np.float32([[-0.1517,-0.9669,0.2050]])
v2_B = np.float32([[-0.8285,0.5522,-0.0955]])
v2_N = np.float32([[-0.8393,0.4494,-0.3044]])

r = OLAE(v1_B, v1_N, v2_B, v2_N)
r = EP2DCM(r)

print(r)