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

def tilde2(m):
	r = np.float32([[0, -m[0, 2], m[0, 1]],
				  [m[0, 2], 0, -m[0, 0]],
				  [-m[0, 1], m[0, 0], 0]])

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

def EP2DCM(q):
	q = q / np.linalg.norm(q)

	g = np.float32([[q[0, 1]/q[0, 0], q[0, 2]/q[0, 0], q[0, 3]/q[0, 0]]]).T
	G = tilde2(g.T)

	C = (1 - np.dot(g.T, g)) * np.eye(3) + 2 * np.dot(g, g.T) - 2 * G
	C = C / (1 + np.dot(g.T, g))

	return C

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

def MRP2DCM(s):
	norm = np.dot(s.T, s)
	_tilde = tilde(s)

	C = 8 * np.dot(_tilde, _tilde) - 4 * (1 - norm) * _tilde
	C = C / np.square(1 + norm)
	C = np.eye(3) + C

	return C

def DCM2MRP(m):
	trace = m[0, 0] + m[1, 1] + m[2, 2]
	S = sqrt(trace + 1)

	s = np.float32([[m[1, 2] - m[2, 1]],
					[m[2, 0] - m[0, 2]],
					[m[0, 1] - m[1, 0]]])

	s = s / (S * (S + 2))

	return s

def MRPAddition(s_p, s_pp):
	s = (1 - np.sum(np.square(s_p))) * s_pp + (1 - np.sum(np.square(s_pp))) * s_p
	s = s - 2 * np.cross(s_pp, s_p)
	s = s / (1 + np.sum(np.square(s_p)) * np.sum(np.square(s_pp)) - 2 * np.dot(s_p, s_pp.T))

	return s

def MRPSubtraction(s, s_p):
	s_pp = (1 - np.sum(np.square(s_p))) * s - (1 - np.sum(np.square(s))) * s_p
	s_pp = s_pp + 2 * np.cross(s, s_p)
	s_pp = s_pp / (1 + np.sum(np.square(s_p)) * np.sum(np.square(s)) + 2 * np.dot(s_p, s.T))

	return s_pp

def MRPNormalization(s):
	if np.sqrt(np.sum(np.square(s))) > 1:
		s = s * (-1 / np.sum(np.square(s)))

	return s

def MRPDiffEq(s, w):
	ds = (1 - np.sum(np.square(s))) * np.eye(3) + 2 * tilde2(s) + 2 * np.multiply(s, s.T)
	ds = ds / 4
	ds = np.dot(ds, w)

	print(ds)

	return ds

def DCM2PRV(m):
	cos_phi = 0.5 * (m[0, 0] + m[1, 1] + m[2, 2] - 1)
	phi = math.acos(cos_phi)

	e = np.float32([[m[1,2] - m[2,1], m[2,0] - m[0, 2], m[0,1] - m[1,0]]])
	e = e / (2 * sin(phi))

	return e, phi

def TriadFrame(v1, v2):
	t1 = v1
	t2 = np.cross(v1, v2)
	t2 = t2 / np.linalg.norm(t2)
	t3 = np.cross(t1, t2)

	Triad_frame = np.float32([t1, t2, t3])

	return Triad_frame

def Triad_BdashN(v1_N, v1_B, v2_N, v2_B):
	N_Triad = TriadFrame(v1_N, v2_N).T.reshape((3, 3))
	B_Triad = TriadFrame(v1_B, v2_B).T.reshape((3, 3))

	BdashN = np.dot(B_Triad, N_Triad.T)

	return BdashN

def TriadError(BdashN, BN):
	BdashB = np.dot(BdashN, BN.T)

	e, phi = DCM2PRV(BdashB)

	return phi #In radians

def Devenport_q(v1_B, v1_N, v2_B, v2_N, w1 = 1, w2 = 1):
	B = w1 * np.dot(v1_B.T, v1_N) + w2 * np.dot(v2_B.T, v2_N)
	sigma = B[0, 0] + B[1, 1] + B[2, 2]
	Z = np.float32([[B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]]])
	S = B + B.T
	S_p = S - sigma * np.eye(3)
	K = np.float32([[sigma, Z[0, 0], Z[0, 1], Z[0, 2]],
					[Z[0, 0], S_p[0, 0], S_p[0, 1], S_p[0, 2]],
					[Z[0, 1], S_p[1, 0], S_p[1, 1], S_p[1, 2]],
					[Z[0, 2], S_p[2, 0], S_p[2, 1], S_p[2, 2]]])

	vals, vecs = np.linalg.eig(K)
	max_index = np.argmax(vals)

	lambd = vals[max_index]
	eig_vec = vecs[max_index]

	C = EP2DCM(eig_vec.reshape((1, 4)))

	return C

def QUEST(v1_B, v1_N, v2_B, v2_N, w1 = 1, w2 = 1):
	lambda_optimal = w1 + w2
	B = w1 * np.dot(v1_B.T, v1_N) + w2 * np.dot(v2_B.T, v2_N)
	S = B + B.T
	sigma = B[0, 0] + B[1, 1] + B[2, 2]
	Z = np.float32([[B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]]])

	q_dash = np.dot(np.linalg.inv((lambda_optimal + sigma) * np.eye(3) - S), Z.T)

	B_dash = (1 / (1 + np.dot(q_dash.T, q_dash))) * np.float32([[1, q_dash[0], q_dash[1], q_dash[2]]])

	return B_dash

def OLAE(v1_B, v1_N, v2_B, v2_N, w1 = 1, w2 = 1):
	S = np.float32([[tilde2(v1_B + v1_N), tilde2(v2_B + v2_N)]]).reshape((6, 3))
	d = np.float32([[v1_B - v1_N, v2_B - v2_N]]).reshape((6, 1))
	W = np.eye(6)
	q_dash = np.linalg.inv(np.dot(np.dot(S.T, W), S))
	q_dash = np.dot(q_dash, S.T)
	q_dash = np.dot(q_dash, W)
	q_dash = np.dot(q_dash, d)

	B_dash = (1 / (1 + np.dot(q_dash.T, q_dash))) * np.float32([[1, q_dash[0], q_dash[1], q_dash[2]]])

	return B_dash