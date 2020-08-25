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

B_N = np.float32([[0.969846,-0.200706,-0.138258],
				[0.17101,0.96461,-0.200706],
				[0.173648,0.17101,0.969846]]).T

BN = np.float32([[0.963592,-0.223042,-0.147454],
				[0.187303,0.956645,-0.223042],
				[0.190809,0.187303,0.963592]]).T

r = TriadError(B_N, BN)

print(r)