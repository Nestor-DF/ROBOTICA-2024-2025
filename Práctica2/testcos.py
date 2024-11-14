import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

E = np.array([15, 0])
R = np.array([10, 0])
print("\nE: ", E)
print("\nR: ", R)

v1 = np.array(E - R)
print("\nv1: ", v1)
v2 = np.array(np.array([10, 10]) - R)
print("\nv2: ", v2)

v1 = np.array(v1 / np.linalg.norm(v1))
print("\nv1: ", v1)
v2 = np.array(v2 / np.linalg.norm(v2))
print("\nv2: ", v2)

cos_alpha = np.dot(v1, v2)

print("\ncos_alpha: ", cos_alpha)
print("\ndegrees(acos(cos_alpha)): ", degrees(acos(cos_alpha)))
