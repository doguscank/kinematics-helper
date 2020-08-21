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

B_0 = np.float32([0.408248,0.,0.408248,0.816497])

r = FirstOrderQuaternionIntegrator(B_0, Week3ConceptCheck8, 42, 0.001)
norm = beta_norm((r[0], r[1], r[2], r[3]))

print(r, norm)