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

#M = GetDCM('321', [-10, 10, 5], rad = False)

#R = np.float32([[-0.5, 0.5, 0.25]])

#R_B = np.dot(M, R.T)

I_c = np.float32([[10, 1, -1], [1, 5, 1], [-1, 1, 8]])

w_B = np.float32([[0.01, -0.01, 0.01]])

#I_p = 12.5 * np.dot(tilde2(R_B.T), tilde2(R_B.T).T)

#I_o = I_c + I_p

#sigma_DB = np.float32([[0.1, 0.2, 0.3]])

#M = MRP2DCM(sigma_DB.T)

#I_c_D = np.dot(np.dot(M, I_c), M.T)

#print(I_c_D)

#values, vectors = np.linalg.eig(I_c)

#vectors[1] *= -1

r = 0.5 * np.dot(np.dot(w_B, I_c), w_B.T)

print(r)