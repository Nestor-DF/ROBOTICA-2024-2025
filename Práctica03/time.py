#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import *
from robot import robot
import numpy as np
import time
import matplotlib.pyplot as plt


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


def mostrar(objetivos, tray_ideal, tray_real):
    plt.ion()  # modo interactivo
    objT = np.array(objetivos).T.tolist()
    trayT = np.array(tray_real).T.tolist()
    ideT = np.array(tray_ideal).T.tolist()
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
    tray_idealT = np.array(tray_ideal).T.tolist()
    plt.plot(tray_idealT[0], tray_idealT[1], "-g", label="Trayectoria Ideal")
    plt.plot(tray_real[0][0], tray_real[0][1], "or")
    r = radio * 0.1
    for p in tray_real:
        plt.plot(
            [p[0], p[0] + r * cos(p[2])],
            [p[1], p[1] + r * sin(p[2])],
            "-r",
            label="Trayectoria Real" if p == tray_real[0] else "",
        )
    objT = np.array(objetivos).T.tolist()
    plt.plot(objT[0], objT[1], "-.o", label="Objetivos")
    plt.legend(loc="upper right")
    plt.show()
    input()
    plt.clf()


def localizacion(balizas, real, ideal, centro, radio, mostrar=0):
    # Buscar la localización más probable del robot, a partir de su sistema
    # sensorial, dentro de una región cuadrada de centro "centro" y lado "2*radio".
    imagen = []
    mejor_error = float("inf")
    mejor_punto = [ideal.x, ideal.y]
    incremento = 0.05

    for i in np.arange(-radio, radio + incremento, incremento):
        fila = []
        for j in np.arange(-radio, radio + incremento, incremento):
            ideal.set(centro[0] + j, centro[1] + i, ideal.orientation)
            error = ideal.measurement_prob(real.sense(balizas), balizas)
            fila.append(error)
            if error < mejor_error:
                mejor_error = error
                mejor_punto = [centro[0] + j, centro[1] + i]
        imagen.append(fila)

    mejor_orientacion = ideal.orientation
    for angulo in np.linspace(-pi, pi, 72):  # Dividir el rango en 72 pasos (5º)
        ideal.set(mejor_punto[0], mejor_punto[1], angulo)
        error = ideal.measurement_prob(real.sense(balizas), balizas)
        if error < mejor_error:
            mejor_error = error
            mejor_orientacion = angulo

    ideal.set(mejor_punto[0], mejor_punto[1], mejor_orientacion)

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
        plt.plot(ideal.x, ideal.y, "D", c="#ff00ff", ms=10, mew=2, label="Robot ideal")
        plt.plot(real.x, real.y, "D", c="#00ff00", ms=10, mew=2, label="Robot real")
        plt.legend(loc="upper right")
        plt.show()
        input()
        plt.clf()


def localizacion_piramidal(balizas, real, ideal, centro, radio):
    mejor_error = float("inf")
    mejor_punto = [ideal.x, ideal.y]

    niveles = 3  # Número de niveles en la búsqueda piramidal
    for nivel in range(niveles):
        incremento = radio / (2**nivel)  # Reducir el paso en cada nivel
        rango = np.arange(-radio, radio + incremento, incremento)
        for i in rango:
            for j in rango:
                ideal.set(centro[0] + j, centro[1] + i, ideal.orientation)
                error = ideal.measurement_prob(real.sense(balizas), balizas)
                if error < mejor_error:
                    mejor_error = error
                    mejor_punto = [centro[0] + j, centro[1] + i]
        # Ajustar el centro para el siguiente nivel al mejor encontrado
        centro = [mejor_punto[0], mejor_punto[1]]

    mejor_orientacion = ideal.orientation
    for angulo in np.linspace(-pi, pi, 72):  # Dividir el rango en 72 pasos (5º)
        ideal.set(mejor_punto[0], mejor_punto[1], angulo)
        error = ideal.measurement_prob(real.sense(balizas), balizas)
        if error < mejor_error:
            mejor_error = error
            mejor_orientacion = angulo

    ideal.set(mejor_punto[0], mejor_punto[1], mejor_orientacion)


def calcular_radio(objetivos, centro):
    distancias = [distancia(centro, obj) for obj in objetivos]
    radio = max(distancias) + 1  # margen de seguridad
    return radio


def calcular_centro(puntos):
    if not puntos:
        raise ValueError("La lista de puntos no puede estar vacía")
    x_centro = sum(p[0] for p in puntos) / len(puntos)
    y_centro = sum(p[1] for p in puntos) / len(puntos)
    return [x_centro, y_centro]


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
EPSILON = 0.02  # Umbral de distancia
# Si es grande localizas menos veces pero la región es más grande (porque hay más error) y viceversa
THRESHOLD = (
    0.2  # Umbral de desviación entre robot real vs ideal (0.5 es relativamente grande)
)
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
localiza = 0
no_localiza = 0
error_grande = 0

objetivos_ampliados = objetivos + [[ideal.x, ideal.y]]
centro = calcular_centro(objetivos_ampliados)
radio = calcular_radio(objetivos_ampliados, centro)
print("Centro: ", centro)
print("Radio: ", radio)
localizacion(objetivos, real, ideal, centro, radio, 0)

start_time = time.time()

for punto in objetivos:
    # len(tray_ideal) <= 1000: -> por si te pierdes mucho y no encuentras las balizas
    while distancia(tray_ideal[-1], punto) > EPSILON and len(tray_ideal) <= 1000:
        pose = ideal.pose()

        w = angulo_rel(pose, punto)
        # Acotar respecto a la constante que determina los máximos
        if w > W:
            w = W
        if w < -W:
            w = -W

        v = distancia(pose, punto)
        # Acotar respecto a la constante que determina los máximos
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

        error = ideal.measurement_prob(real.sense(objetivos), objetivos)
        if error > THRESHOLD:
            localizacion_piramidal(objetivos, real, ideal, [ideal.x, ideal.y], 2 * error)

        espacio += v
        tiempo += 1

# Fin del bucle principal
end_time = time.time()

# Calcular y mostrar el tiempo de ejecución
execution_time = end_time - start_time
print(f"Tiempo de ejecución del bucle principal: {execution_time:.4f} segundos")
