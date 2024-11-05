#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática directa mediante Denavit-Hartenberg.

# Ejemplo:
# ./cdDH.py 30 45

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ******************************************************************************
# Declaración de funciones


def ramal(I, prev=[], base=0):
    # Convierte el robot a una secuencia de puntos para representar
    O = []
    if I:
        if isinstance(I[0][0], list):
            for j in range(len(I[0])):
                O.extend(ramal(I[0][j], prev, base or j < len(I[0]) - 1))
        else:
            O = [I[0]]
            O.extend(ramal(I[1:], I[0], base))
            if base:
                O.append(prev)
    return O


def muestra_robot(O, ef=[]):
    # Pinta en 3D
    OR = ramal(O)
    OT = np.array(OR).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Bounding box cúbico para simular el ratio de aspecto correcto
    max_range = np.array(
        [
            OT[0].max() - OT[0].min(),
            OT[1].max() - OT[1].min(),
            OT[2].max() - OT[2].min(),
        ]
    ).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
        OT[0].max() + OT[0].min()
    )
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
        OT[1].max() + OT[1].min()
    )
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
        OT[2].max() + OT[2].min()
    )
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], "w")
    ax.plot3D(OT[0], OT[1], OT[2], marker="s")
    ax.plot3D([0], [0], [0], marker="o", color="k", ms=10)
    if not ef:
        ef = OR[-1]
    ax.plot3D([ef[0]], [ef[1]], [ef[2]], marker="s", color="r")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    return


def arbol_origenes(O, base=0, sufijo=""):
    # Da formato a los origenes de coordenadas para mostrarlos por pantalla
    if isinstance(O[0], list):
        for i in range(len(O)):
            if isinstance(O[i][0], list):
                for j in range(len(O[i])):
                    arbol_origenes(O[i][j], i + base, sufijo + str(j + 1))
            else:
                print(
                    "(O"
                    + str(i + base)
                    + sufijo
                    + ")0\t= "
                    + str([round(j, 3) for j in O[i]])
                )
    else:
        print("(O" + str(base) + sufijo + ")0\t= " + str([round(j, 3) for j in O]))


def muestra_origenes(O, final=0):
    # Muestra los orígenes de coordenadas para cada articulación
    print("Orígenes de coordenadas:")
    arbol_origenes(O)
    if final:
        print("E.Final = " + str([round(j, 3) for j in final]))


def matriz_T(d, theta, a, alpha):
    # Calcula la matriz T (ángulos de entrada en grados)
    th = theta * pi / 180
    al = alpha * pi / 180
    return [
        [cos(th), -sin(th) * cos(al), sin(th) * sin(al), a * cos(th)],
        [sin(th), cos(th) * cos(al), -sin(al) * cos(th), a * sin(th)],
        [0, sin(al), cos(al), d],
        [0, 0, 0, 1],
    ]


# ******************************************************************************


# Introducción de los valores de las articulaciones
nvar = 7  # Número de variables
if len(sys.argv) != nvar + 1:
    sys.exit("El número de articulaciones no es el correcto (" + str(nvar) + ")")
p = [float(i) for i in sys.argv[1 : nvar + 1]]

# Parámetros D-H:
d = [5, 2, p[2], 2, 0, 0, 5, 0, 0, 1, 1, 1]
th = [90 + p[0], p[1], -90, p[3], p[4], 90, p[5], 0, 0, 0, 0, 0]
a = [0, 0, 0, 0, 3, 0, 0, p[6], -p[6], 0, 0, 0]
al = [90, -90, 0, 90, 0, 90, 0, 0, 0, 0, 0, 0]

# Orígenes para cada articulación
o00 = [0, 0, 0, 1]


# Cálculo matrices transformación
T0_1 = matriz_T(d[0], th[0], a[0], al[0])
T1_2 = matriz_T(d[1], th[1], a[1], al[1])
T2_3 = matriz_T(d[2], th[2], a[2], al[2])
T3_4 = matriz_T(d[3], th[3], a[3], al[3])
T4_5p = matriz_T(d[4], th[4], a[4], al[4])
T5p_5 = matriz_T(d[5], th[5], a[5], al[5])
T5_6 = matriz_T(d[6], th[6], a[6], al[6])
T6_71p = matriz_T(d[7], th[7], a[7], al[7])
T6_72p = matriz_T(d[8], th[8], a[8], al[8])
T71p_71 = matriz_T(d[9], th[9], a[9], al[9])
T72p_72 = matriz_T(d[10], th[10], a[10], al[10])
T6_EF = matriz_T(d[11], th[11], a[11], al[11])

T0_2 = np.dot(T0_1, T1_2)
T0_3 = np.dot(T0_2, T2_3)
T0_4 = np.dot(T0_3, T3_4)
T0_5p = np.dot(T0_4, T4_5p)
T0_5 = np.dot(T0_5p, T5p_5)
T0_6 = np.dot(T0_5, T5_6)
T0_71p = np.dot(T0_6, T6_71p)
T0_72p = np.dot(T0_6, T6_72p)
T0_71 = np.dot(T0_71p, T71p_71)
T0_72 = np.dot(T0_72p, T72p_72)
T0_EF = np.dot(T0_6, T6_EF)


# Transformación de cada articulación
o110 = np.dot(T0_1, o00).tolist()
o220 = np.dot(T0_2, o00).tolist()
o330 = np.dot(T0_3, o00).tolist()
o440 = np.dot(T0_4, o00).tolist()
o55p0 = np.dot(T0_5p, o00).tolist()
o550 = np.dot(T0_5, o00).tolist()
o660 = np.dot(T0_6, o00).tolist()
o771p0 = np.dot(T0_71p, o00).tolist()
o772p0 = np.dot(T0_72p, o00).tolist()
o7710 = np.dot(T0_71, o00).tolist()
o7720 = np.dot(T0_72, o00).tolist()
oEF0 = np.dot(T0_EF, o00).tolist()


# Mostrar resultado de la cinemática directa
muestra_origenes(
    [
        o00,
        o110,
        o220,
        o330,
        o440,
        o55p0,
        o550,
        o660,
        [[o771p0, o7710], [o772p0, o7720]],
    ],
    oEF0,
)
muestra_robot(
    [
        o00,
        o110,
        o220,
        o330,
        o440,
        o55p0,
        o550,
        o660,
        [[o771p0, o7710], [o772p0, o7720]],
    ],
    oEF0,
)
input()
