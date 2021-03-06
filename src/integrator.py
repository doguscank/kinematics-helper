import numpy as np
from transform import *

def FirstOrderIntegrator(X_0, func, end_time, time_step):
	X = X_0

	for t in np.arange(0.0, end_time, time_step):
		X = X + func(t) * time_step

	return X

def FirstOrderQuaternionIntegrator(B_0, func, end_time, time_step):
	B = B_0

	for t in np.arange(0.0, end_time, time_step):
		B_diff = B_diff_eq((B[0], B[1], B[2], B[3]), func(t))
		B = B + B_diff * time_step

	return B

def FirstOrderCRPIntegrator(q_0, func, end_time, time_step):
	q = q_0

	for t in np.arange(0.0, end_time, time_step):
		q = q + func(t, q) * time_step

	return q

def FirstOrderMRPIntegrator(s_0, func, end_time, time_step):
	s = s_0

	for t in np.arange(0.0, end_time, time_step):
		s = s + MRPDiffEq(s, func(t)) * time_step
		s = MRPNormalization(s)

	return s