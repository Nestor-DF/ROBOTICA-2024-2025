#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Rob�tica Computacional
# Grado en Ingenier�a Inform�tica (Cuarto)
# Pr�ctica 5:
#     Simulaci�n de robots m�viles holon�micos y no holon�micos.

# localizacion.py

import sys
from math import *
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ******************************************************************************
# Declaraci�n de funciones


def distancia(a, b):
    # Distancia entre dos puntos (admite poses)
    return np.linalg.norm(np.subtract(a[:2], b[:2]))


def angulo_rel(pose, p):
    # Diferencia angular entre una pose y un punto objetivo 'p'
    w = atan2(p[1] - pose[1], p[0] - pose[0]) - pose[2]
    while w > pi:
        w -= 2 * pi
    while w < -pi:
        w += 2 * pi
    return w


def mostrar(objetivos, ideal, trayectoria):
    # Mostrar objetivos y trayectoria:
    # plt.ion() # modo interactivo
    # Fijar los bordes del gr�fico
    objT = np.array(objetivos).T.tolist()
    trayT = np.array(trayectoria).T.tolist()
    ideT = np.array(ideal).T.tolist()
    bordes = [
        min(trayT[0] + objT[0] + ideT[0]),
        max(trayT[0] + objT[0] + ideT[0]),
        min(trayT[1] + objT[1] + ideT[1]),
        max(trayT[1] + objT[1] + ideT[1]),
    ]
    centro = [(bordes[0] + bordes[1]) / 2.0, (bordes[2] + bordes[3]) / 2.0]
    radio = max(bordes[1] - bordes[0], bordes[3] - bordes[2]) * 0.75
    plt.xlim(centro[0] - radio, centro[0] + radio)
    plt.ylim(centro[1] - radio, centro[1] + radio)
    # Representar objetivos y trayectoria
    idealT = np.array(ideal).T.tolist()
    plt.plot(idealT[0], idealT[1], "-g")
    plt.plot(trayectoria[0][0], trayectoria[0][1], "or")
    r = radio * 0.1
    for p in trayectoria:
        plt.plot([p[0], p[0] + r * cos(p[2])], [p[1], p[1] + r * sin(p[2])], "-r")
        # plt.plot(p[0],p[1],'or')
    objT = np.array(objetivos).T.tolist()
    plt.plot(objT[0], objT[1], "-.o")
    plt.show()
    input()
    plt.clf()


def localizacion(balizas, real, ideal, centro, radio, mostrar=0):
    # Buscar la localizaci�n m�s probable del robot, a partir de su sistema
    # sensorial, dentro de una regi�n cuadrada de centro "centro" y lado "2*radio".
    # La imagen que almacenará todos los errores dados para todos los puntos en el radio
    imagen = []
    min_error = sys.float_info.max
    mejorpunto = []
    # Incremento para recorrer todo el radio
    incremento = 0.05
    for y in np.arange(-radio, radio, incremento):
        imagen.append([])
        for x in np.arange(-radio, radio, incremento):
            ideal.set(centro[0] + x, centro[1] + y, ideal.orientation)
            error = real.measurement_prob(ideal.sense(balizas), balizas)
            imagen[-1].append(error)
            if error < min_error:
                min_error = error
                mejorpunto = [centro[0] + x, centro[1] + y]

    ideal.set(mejorpunto[0], mejorpunto[1], real.orientation)

    mejor_orientacion = sys.float_info.max
    mejorOrientacion = ideal.orientation
    for i in np.arange(-pi, pi, 0.01):
        ideal.set(mejorpunto[0], mejorpunto[1], i)
        error = real.measurement_prob(ideal.sense(balizas), balizas)
        if error < mejor_orientacion:
            mejor_orientacion = error
            mejorOrientacion = i
    if mejorOrientacion:
        ideal.set(mejorpunto[0], mejorpunto[1], mejorOrientacion)

    if mostrar:
        plt.ion()  # modo interactivo
        plt.xlim(centro[0] - radio, centro[0] + radio)
        plt.ylim(centro[1] - radio, centro[1] + radio)
        imagen.reverse()
        plt.imshow(
            imagen,
            extent=[
                centro[0] - radio,
                centro[0] + radio,
                centro[1] - radio,
                centro[1] + radio,
            ],
        )
        balT = np.array(balizas).T.tolist()
        plt.plot(balT[0], balT[1], "or", ms=10)
        plt.plot(ideal.x, ideal.y, "D", c="#ff00ff", ms=10, mew=2)
        plt.plot(real.x, real.y, "D", c="#00ff00", ms=10, mew=2)
        plt.show()
        input()
        plt.clf()


def localizacion_piramidal(
    balizas, real, ideal, centro, radio, mostrar=0, niveles=3, factor=2
):
    """
    Localización jerárquica utilizando búsqueda piramidal.

    Args:
        balizas: Lista de balizas.
        real: Robot real.
        ideal: Robot ideal.
        centro: Centro de búsqueda [x, y].
        radio: Radio inicial de búsqueda.
        mostrar: Bandera para mostrar la visualización.
        niveles: Número de niveles de refinamiento.
        factor: Factor de reducción del paso en cada nivel.
    """
    mejor_punto = centro
    mejor_error = sys.float_info.max

    # Búsqueda jerárquica
    for nivel in range(niveles):
        paso = radio / (factor**nivel)  # Ajustar el paso en cada nivel
        rango = np.arange(-radio, radio, paso)
        for y in rango:
            for x in rango:
                # Evaluar la posición actual
                ideal.set(centro[0] + x, centro[1] + y, ideal.orientation)
                error = real.measurement_prob(ideal.sense(balizas), balizas)
                if error < mejor_error:
                    mejor_error = error
                    mejor_punto = [centro[0] + x, centro[1] + y]

        # Actualizar centro y reducir radio para el siguiente nivel
        centro = mejor_punto
        radio /= factor

    # Refinar orientación
    mejor_orientacion = sys.float_info.max
    mejor_ang = ideal.orientation
    for ang in np.arange(-pi, pi, 0.01):
        ideal.set(mejor_punto[0], mejor_punto[1], ang)
        error = real.measurement_prob(ideal.sense(balizas), balizas)
        if error < mejor_orientacion:
            mejor_orientacion = error
            mejor_ang = ang

    # Establecer la mejor pose
    ideal.set(mejor_punto[0], mejor_punto[1], mejor_ang)

    # Visualización opcional
    if mostrar:
        plt.ion()  # Modo interactivo
        plt.xlim(centro[0] - radio, centro[0] + radio)
        plt.ylim(centro[1] - radio, centro[1] + radio)
        plt.plot(ideal.x, ideal.y, "D", c="#ff00ff", ms=10, mew=2)
        plt.plot(real.x, real.y, "D", c="#00ff00", ms=10, mew=2)
        plt.show()
        input()
        plt.clf()


# ******************************************************************************

# Definici�n del robot:
P_INICIAL = [0.0, 4.0, 0.0]  # Pose inicial (posici�n y orientacion)
V_LINEAL = 0.7  # Velocidad lineal    (m/s)
V_ANGULAR = 140.0  # Velocidad angular   (�/s)
FPS = 10.0  # Resoluci�n temporal (fps)

HOLONOMICO = 1
GIROPARADO = 0
LONGITUD = 0.2

# Definici�n de trayectorias:
trayectorias = [
    [[1, 3]],
    [[0, 2], [4, 2]],
    [[2, 4], [4, 0], [0, 0]],
    [[2, 4], [2, 0], [0, 2], [4, 2]],
    [[2 + 2 * sin(0.8 * pi * i), 2 + 2 * cos(0.8 * pi * i)] for i in range(5)],
]

# Definición de los puntos objetivo:
if len(sys.argv) < 2 or int(sys.argv[1]) < 0 or int(sys.argv[1]) >= len(trayectorias):
    sys.exit(sys.argv[0] + " <indice entre 0 y " + str(len(trayectorias) - 1) + ">")
objetivos = trayectorias[int(sys.argv[1])]

# Definici�n de constantes:
EPSILON = 0.1  # Umbral de distancia
V = V_LINEAL / FPS  # Metros por fotograma
W = V_ANGULAR * pi / (180 * FPS)  # Radianes por fotograma
THRESHOLD = 0.1

ideal = robot()
ideal.set_noise(0, 0, 0.1)  # Ruido lineal / radial / de sensado
ideal.set(*P_INICIAL)  # operador 'splat'

real = robot()
real.set_noise(0.01, 0.01, 0.1)  # Ruido lineal / radial / de sensado
real.set(*P_INICIAL)

tray_ideal = [ideal.pose()]  # Trayectoria percibida
tray_real = [real.pose()]  # Trayectoria seguida

tiempo = 0.0
espacio = 0.0
random.seed(0)

localizacion(objetivos, real, ideal, [2, 2], 3, 1)

# random.seed(int(datetime.now().timestamp()))
for punto in objetivos:
    while distancia(tray_ideal[-1], punto) > EPSILON and len(tray_ideal) <= 1000:
        pose = ideal.pose()

        w = angulo_rel(pose, punto)
        if w > W:
            w = W
        if w < -W:
            w = -W
        v = distancia(pose, punto)
        if v > V:
            v = V
        if v < 0:
            v = 0

        if HOLONOMICO:
            if GIROPARADO and abs(w) > 0.01:
                v = 0
            ideal.move(w, v)
            real.move(w, v)
        else:
            ideal.move_triciclo(w, v, LONGITUD)
            real.move_triciclo(w, v, LONGITUD)
        tray_ideal.append(ideal.pose())
        tray_real.append(real.pose())

        weight = real.measurement_prob(ideal.sense(objetivos), objetivos)

        if weight > THRESHOLD:
            # localizacion(objetivos, real, ideal, ideal.pose(), 2 * weight, 0)
            localizacion_piramidal(objetivos, real, ideal, ideal.pose(), 2 * weight)

        espacio += v
        tiempo += 1

if len(tray_ideal) > 1000:
    print(
        "<!> Trayectoria muy larga - puede que no se haya alcanzado la posicion final."
    )
print("Recorrido: " + str(round(espacio, 3)) + "m / " + str(tiempo / FPS) + "s")
print(
    "Distancia real al objetivo: "
    + str(round(distancia(tray_real[-1], objetivos[-1]), 3))
    + "m"
)
mostrar(objetivos, tray_ideal, tray_real)  # Representaci�n gr�fica
