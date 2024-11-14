#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional -
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ******************************************************************************
# Declaración de funciones


def muestra_origenes(O, final=0):
    print("Origenes de coordenadas:")
    for i in range(len(O)):
        print("(O" + str(i) + ")0\t= " + str([round(j, 3) for j in O[i]]))
    if final:
        print("E.Final = " + str([round(j, 3) for j in final]))


def muestra_robot(O, obj):
    plt.figure()
    plt.xlim(-L, L)
    plt.ylim(-L, L)
    T = [np.array(o).T.tolist() for o in O]
    for i in range(len(T)):
        plt.plot(T[i][0], T[i][1], "-o", color=cs.hsv_to_rgb(i / float(len(T)), 1, 1))
    plt.plot(obj[0], obj[1], "*")
    plt.pause(0.0001)
    plt.show()
    input()
    plt.close()


def matriz_T(d, th, a, al):
    return [
        [cos(th), -sin(th) * cos(al), sin(th) * sin(al), a * cos(th)],
        [sin(th), cos(th) * cos(al), -sin(al) * cos(th), a * sin(th)],
        [0, sin(al), cos(al), d],
        [0, 0, 0, 1],
    ]


def cin_dir(th, a):
    # Sea 'th' el vector de thetas
    # Sea 'a'  el vector de longitudes
    T = np.identity(4)
    o = [[0, 0]]
    for i in range(len(th)):
        T = np.dot(T, matriz_T(0, th[i], a[i], 0))
        tmp = np.dot(T, [0, 0, 0, 1])
        o.append([tmp[0], tmp[1]])
    return o


# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# valores articulares arbitrarios para la cinemática directa inicial
th = [0.0, 0.0, 0.0]
a = [5.0, 5.0, 5.0]
L = sum(a)  # variable para representación gráfica
EPSILON = 0.01

plt.ion()  # modo interactivo

# introducción del punto para la cinemática inversa
if len(sys.argv) != 3:
    sys.exit("python " + sys.argv[0] + " x y")
objetivo = [float(i) for i in sys.argv[1:]]
O = cin_dir(th, a)
# O = zeros(len(th) + 1)  # Reservamos estructura en memoria
# Calculamos la posicion inicial
print("- Posicion inicial:")
muestra_origenes(O)

dist = float("inf")
prev = 0.0
iteracion = 1

# Mientras no llegues al punto destino o no puedas acercarte más
while dist > EPSILON and abs(prev - dist) > EPSILON / 100.0:
    prev = dist
    O = [cin_dir(th, a)]
    # Para cada combinación de articulaciones:
    for i in range(len(th)):
        # Parte del código a resolver: cálculo de la cinemática inversa
        E = np.array(O[-1][-1 - i])
        R = np.array(O[-1][-1 - i - 1])
        # Definir dos vectores
        v1 = E - R
        v2 = objetivo - R
        # Paso 1: Normalizar los vectores
        if np.linalg.norm(v1) != 0:
            v1 = v1 / np.linalg.norm(v1)
        else:
            v1 = np.zeros_like(v1)
        if np.linalg.norm(v2) != 0:
            v2 = v2 / np.linalg.norm(v2)
        else:
            v2 = np.zeros_like(v2)
        # Paso 2: Calcular el ángulo utilizando el coseno
        cos_alpha = np.dot(v1, v2)  # Producto punto
        cos_alpha = np.clip(
            cos_alpha, -1.0, 1.0
        )  # Limitar para evitar problemas numéricos
        alpha_magnitude = acos(cos_alpha)
        # Paso 3: Determinar la dirección del ángulo usando el producto cruzado
        # En 2D, el producto cruzado es un escalar y puede dar positivo o negativo
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]  # Producto cruzado escalar en 2D
        # Si el producto cruzado es positivo, el ángulo es positivo; si es negativo, el ángulo es negativo
        if cross_product < 0:
            alpha = -alpha_magnitude
        else:
            alpha = alpha_magnitude
        th[-1 - i] = th[-1 - i] + alpha
        O.append(cin_dir(th, a))
    dist = np.linalg.norm(np.subtract(objetivo, O[-1][-1]))
    print("\n- Iteracion " + str(iteracion) + ":")
    muestra_origenes(O[-1])
    muestra_robot(O, objetivo)
    print("Distancia al objetivo = " + str(round(dist, 5)))
    iteracion += 1
    O[0] = O[-1]

if dist <= EPSILON:
    print("\n" + str(iteracion) + " iteraciones para converger.")
else:
    print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print("- Umbral de convergencia epsilon: " + str(EPSILON))
print("- Distancia al objetivo:          " + str(round(dist, 5)))
print("- Valores finales de las articulaciones:")
for i in range(len(th)):
    print("  theta" + str(i + 1) + " = " + str(round(th[i], 3)))
for i in range(len(th)):
    print("  L" + str(i + 1) + "     = " + str(round(a[i], 3)))
