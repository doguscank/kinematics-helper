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

q_0 = np.float32([[0.4, 0.2, -0.1]])

r = FirstOrderCRPIntegrator(q_0, Week3ConceptCheck13, 42, 0.001)

print(r)
print(CRPNorm(r))