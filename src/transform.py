import numpy as np
from math import sin, cos, atan2, sqrt, pow
import math

def M1(d):
	M1 = np.float32([[1.0, 0.0, 0.0], 
					 [0.0, cos(d), sin(d)],
					 [0.0, -sin(d), cos(d)]])

	return M1

def M2(d):
	M2 = np.float32([[cos(d), 0.0, -sin(d)],
					 [0.0, 1.0, 0.0],
					 [sin(d), 0.0, cos(d)]])

	return M2

def M3(d):
	M3 = np.float32([[cos(d), sin(d), 0.0],
					 [-sin(d), cos(d), 0.0],
					 [0.0, 0.0, 1.0]])

	return M3

def tilde(m):
	r = np.float32([[0, -m[2], m[1]],
				  [m[2], 0, -m[0]],
				  [-m[1], m[0], 0]])

	return r

def deg2rad(d):
	return d * math.pi / 180

def EulerNorm(angles):
	return np.linalg.norm(deg2rad(angles))

def DCM2EP(dcm):
	trace = dcm[0, 0] + dcm[1, 1] + dcm[2, 2]
	B_0_2 = (1 / 4) * (1 + trace)
	B_1_2 = (1 / 4) * (1 + 2 * dcm[0, 0] - trace)	
	B_2_2 = (1 / 4) * (1 + 2 * dcm[1, 1] - trace)
	B_3_2 = (1 / 4) * (1 + 2 * dcm[2, 2] - trace)

	beta_squares = np.float32([[B_0_2, B_1_2, B_2_2, B_3_2]])
	index = np.argmax(beta_squares)

	print(index)

	if index == 0:
		B_0 = sqrt(B_0_2)
		B_1 = (dcm[1, 2] - dcm[2, 1]) / (4 * B_0)
		B_2 = (dcm[2, 0] - dcm[0, 2]) / (4 * B_0)
		B_3 = (dcm[0, 1] - dcm[1, 0]) / (4 * B_0)
	elif index == 1:
		B_1 = sqrt(B_1_2)
		B_0 = (dcm[1, 2] - dcm[2, 1]) / (4 * B_1)
		B_2 = (dcm[0, 1] + dcm[1, 0]) / (4 * B_1)
		B_3 = (dcm[0, 2] + dcm[2, 0]) / (4 * B_1)
	elif index == 2:
		B_2 = sqrt(B_2_2)
		B_1 = (dcm[0, 1] + dcm[1, 0]) / (4 * B_2)
		B_0 = (dcm[2, 0] - dcm[0, 2]) / (4 * B_2)
		B_3 = (dcm[2, 1] + dcm[1, 2]) / (4 * B_2)
	elif index == 3:
		B_3 = sqrt(B_3_2)
		B_1 = (dcm[0, 2] + dcm[2, 0]) / (4 * B_3)
		B_2 = (dcm[2, 1] + dcm[1, 2]) / (4 * B_3)
		B_0 = (dcm[0, 1] - dcm[1, 0]) / (4 * B_3)

	return (B_0, B_1, B_2, B_3)

def B_double_prime(betas):
	B_0, B_1, B_2, B_3 = betas
	B_pp = np.float32([[B_0, -B_1, -B_2, -B_3],
					  [B_1, B_0, B_3, -B_2],
					  [B_2, -B_3, B_0, B_1],
					  [B_3, B_2, -B_1, B_0]])

	return B_pp

def B_diff_eq(betas, omegas):
	B_0, B_1, B_2, B_3 = betas
	B = np.float32([[B_0, -B_1, -B_2, -B_3],
					  [B_1, B_0, -B_3, B_2],
					  [B_2, B_3, B_0, -B_1],
					  [B_3, -B_2, B_1, B_0]])

	dB = 0.5 * np.dot(B, omegas)

	return dB

def EPAddition(B_primes, B_double_primes):
	B_pp = B_double_prime(B_double_primes)
	B_p = np.float32([B_primes])

	r = np.dot(B_pp, B_p.T)

	return r

def beta_norm(betas):
	B_0, B_1, B_2, B_3 = betas
	return sqrt(pow(B_1, 2) + pow(B_2, 2) + pow(B_3, 2))

def beta_norm_np(betas):
	return np.sqrt(np.sum(np.square(betas[1:])))

def CRPAddition(q_p, q_pp):
	q = q_pp + q_p - np.cross(q_pp, q_p)
	q = q / (1 - np.dot(q_pp, q_p.T))

	return q

def CRPSubtraction(q, q_p):
	q_pp = q - q_p + np.cross(q, q_p)
	q_pp = q_pp / (1 + np.dot(q, q_p.T))

	return q_pp

def CRP2DCM(q):
	C = np.multiply((1 - np.dot(q.T, q)), np.eye(3)) + 2 * np.dot(q, q.T) - 2 * tilde(q)
	C = C / (1 + np.dot(q.T, q))

	return C

def DCM2CRP(m):
	trace = m[0, 0] + m[1, 1] + m[2, 2]
	S = sqrt(trace + 1)

	q = np.float32([[m[1, 2] - m[2, 1]],
					[m[2, 0] - m[0, 2]],
					[m[0, 1] - m[1, 0]]])

	q = q / pow(S, 2)

	return q

def CRPNorm(q):
	return np.sqrt(np.sum(np.square(q)))