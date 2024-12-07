#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import *
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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
    plt.ion()  # modo interactivo
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
    # Buscar la localización m�s probable del robot, a partir de su sistema
    # sensorial, dentro de una región cuadrada de centro "centro" y lado "2*radio".
    # imagen = [] # Matriz de probabilidad de dos dimensiones, por cada posición del radio almacenamos el error
    # for (i=-radio, incremento, +radio)
        # for (j=-radio, incremento, +radio)
            # ideal.setPos(centro.x+i, centro.y+j, ideal.orientation)
            # guardar en imagen el cálculo del error (measurement_prob(ideal, real))
            # if (measurement_prob(ideal, real) es mejor que la anterior)
                # almacenamos la posición
    # tenemos la mejor posición 
    # ideal.setPos(mejorPos.x, mejorPos.y, ideal.orientation)

    # Buscar la localización más probable del robot a partir de su sistema sensorial.
    imagen = []  # Matriz de probabilidad de dos dimensiones
    mejor_prob = float("inf")  # Inicializa el mejor error como infinito
    mejor_pos = ideal.pose()  # Inicializa con la posición actual del robot ideal
    incremento = 0.1  # Incremento en la búsqueda en el radio

    for i in np.arange(-radio, radio + incremento, incremento):
        fila = []
        for j in np.arange(-radio, radio + incremento, incremento):
            # Establece temporalmente una nueva posición para el robot ideal
            ideal.set(centro[0] + i, centro[1] + j, ideal.orientation)
            # Calcula la probabilidad de la posición actual
            prob = ideal.measurement_prob(real.sense(balizas), balizas)
            fila.append(prob)
            if prob < mejor_prob:  # Menor error implica mejor posición
                mejor_prob = prob
                mejor_pos = [centro[0] + i, centro[1] + j, ideal.orientation]
        imagen.append(fila)

    # Actualiza la posición del robot ideal con la mejor encontrada
    ideal.set(*mejor_pos)

    # Mostrar el mapa del error si se solicita
    if mostrar:
        plt.ion() # modo interactivo
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


# *******************************************************************************************


# Definición de trayectorias:
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

# Definición del robot:
P_INICIAL = [0.0, 4.0, 0.0]  # Pose inicial (posición y orientacion)
V_LINEAL = 0.7  # Velocidad lineal    (m/s)
V_ANGULAR = 140.0  # Velocidad angular   (�/s)
FPS = 10.0  # Resolución temporal (fps)
HOLONOMICO = 1
GIROPARADO = 0
LONGITUD = 0.2

# Definición de constantes:
EPSILON = 0.1  # Umbral de distancia
V = V_LINEAL / FPS  # Metros por fotograma
W = V_ANGULAR * pi / (180 * FPS)  # Radianes por fotograma

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
# random.seed(int(datetime.now().timestamp()))

# Localizar inicialmente al robot
RADIO = 1.0
localizacion(objetivos, real, ideal, [real.x, real.y], RADIO)

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

        # compara real.sense con ideal.sense
        # si el error es muy grande, localizar (llamar a la funcion localizacion)
        # la funcion localizacion va a calcular cúal es la posición más adecuada para nuestro
        # robot ideal en función de la posición de nuestro robot real
        # es decir, cogemos las medidas de nuestro robot real y las comparamos con todas las posibilidades
        # moviendo nuestro robot ideal virtualmente a ver cúal es el mejor punto (nos lo da measurement_prob)
        # y corregiremos la posición

        # Compara las lecturas del robot real e ideal
        error_mediciones = sum(
            abs(r - i) for r, i in zip(real.sense(objetivos), ideal.sense(objetivos))
        )
        if error_mediciones > 0.5:  # Umbral de error
            # Llama a la función de localización para corregir la posición
            localizacion(objetivos, real, ideal, [real.x, real.y], RADIO)

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
mostrar(objetivos, tray_ideal, tray_real)  # Representación gr�fica
