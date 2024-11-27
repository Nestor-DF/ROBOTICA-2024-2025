#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional -
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import cos, sin, atan2
import numpy as np
import argparse
import json
import matplotlib
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt


def muestra_origenes(O, final=0):
    print("Origenes de coordenadas:")
    for i in range(len(O)):
        print("(O" + str(i) + ")0\t= " + str([round(j, 3) for j in O[i]]))
    if final:
        print("E.Final = " + str([round(j, 3) for j in final]))


figure = plt.figure(figsize=(9, 9))  # Tamaño de la ventana gráfica
plt.ion()  # modo interactivo


def muestra_robot(O, obj, pause_time=1.5, interactive=False):
    # matplotlib.use("TkAgg")  # Para poder configurar la posición de la ventana
    # manager = plt.get_current_fig_manager()
    # manager.window.wm_geometry("-0+0")  # Arriba derecha

    L = sum(a)  # Variable para representación gráfica
    H = 1.5  # Variable para representación gráfica
    plt.xlim(-H * L, H * L)
    plt.ylim(-H * L, H * L)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Visualización del robot")

    # Generar colores distintivos
    num_links = len(O)
    color_palette = list(mcolors.TABLEAU_COLORS.values())
    colors = [color_palette[i % len(color_palette)] for i in range(num_links)]

    # Dibujar
    for i, o in enumerate(O):
        T = np.array(o).T.tolist()
        plt.plot(T[0], T[1], "-o", color=colors[i])
    # Dibujar el punto objetivo
    plt.plot(obj[0], obj[1], "*", color="black", markersize=15, label="Objetivo")

    # Añadir leyenda
    plt.legend(loc="upper right")

    # Mostrar gráfico
    if interactive:
        plt.show()
        input("Presiona Enter para continuar...")
    else:
        plt.pause(pause_time)
    plt.clf()


def matriz_T(d, th, a, al):
    return [
        [cos(th), -sin(th) * cos(al), sin(th) * sin(al), a * cos(th)],
        [sin(th), cos(th) * cos(al), -sin(al) * cos(th), a * sin(th)],
        [0, sin(al), cos(al), d],
        [0, 0, 0, 1],
    ]


def cin_dir(th, a):
    T = np.identity(4)
    o = [[0, 0]]
    for i in range(len(th)):
        T = np.dot(T, matriz_T(0, th[i], a[i], 0))
        tmp = np.dot(T, [0, 0, 0, 1])
        o.append([tmp[0], tmp[1]])
    return o


# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# Configuración de argumentos
parser = argparse.ArgumentParser(description="Programa de cinemática directa e inversa")
parser.add_argument(
    "config_file", type=str, help="Ruta del archivo de configuración JSON"
)
parser.add_argument("x", type=float, help="Coordenada x del punto objetivo")
parser.add_argument("y", type=float, help="Coordenada y del punto objetivo")
parser.add_argument("EPSILON", type=float, help="Umbral de convergencia")
parser.add_argument("--interactive", action="store_true", help="Modo interactivo")

# Parsear argumentos
args = parser.parse_args()

# Leer el archivo de configuración
try:
    with open(args.config_file, "r") as file:
        config = json.load(file)
except FileNotFoundError:
    sys.exit(f"Error: El archivo '{args.config_file}' no se encontró.")
except json.JSONDecodeError as e:
    sys.exit(f"Error: Archivo JSON inválido. {e}")

# Extraer variables de configuración
th = config.get("th", [])
a = config.get("a", [])
prismatica = config.get("prismatica", [])
limites = config.get("limites", [])

# Verificar y lanzar errores si algún valor está fuera de rango
for i in range(len(prismatica)):
    min_range, max_range = limites[i]
    if prismatica[i] == 1:  # Articulación prismática (longitudes)
        if not (0 <= min_range <= max_range):
            raise ValueError(
                f"El rango de a[{i}] = [{min_range}, {max_range}] no está dentro del rango permitido [0, ∞)."
            )
        if not (min_range <= a[i] <= max_range):
            raise ValueError(
                f"El valor de a[{i}] = {a[i]} está fuera del rango permitido [{min_range}, {max_range}]."
            )
        if not (th[i] == 0):
            raise ValueError(
                f"El valor de th[{i}] = {th[i]} debe ser 0 para una articulación prismática."
            )
    else:  # Articulación de revolución (ángulos en grados)
        if not (-180 <= min_range <= max_range <= 180):
            raise ValueError(
                f"El rango de th[{i}] = [{min_range}, {max_range}] no está dentro del rango permitido [-180, 180]."
            )
        if not (min_range <= th[i] <= max_range):
            raise ValueError(
                f"El valor de th[{i}] = {th[i]} está fuera del rango permitido [{np.degrees(min_range)}, {np.degrees(max_range)}]."
            )
        # Convertir los valores de th (en grados) a radianes si es una articulación de revolución
        th[i] = np.radians(th[i])
        # Actualizar el rango convertido a radianes en limites
        limites[i] = [np.radians(min_range), np.radians(max_range)]


# Extraer el punto objetivo de la cinemática inversa, epsilon y modo interactivo de los argumentos
objetivo = [args.x, args.y]
EPSILON = args.EPSILON
interactive = args.interactive

# Mostrar los valores cargados
print("Valores cargados desde el archivo de configuración:")
print(f"th = {th}")
print(f"a = {a}")
print(f"prismatica = {prismatica}")
print(f"limites = {limites}")
print("\nPunto objetivo para la cinemática inversa:")
print(f"objetivo = {objetivo}")

O = cin_dir(th, a)
# O = zeros(len(th) + 1)  # Reservamos estructura en memoria
print("\n- Posicion inicial:")
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
        E = np.array(O[-1][-1])  # Punto final del robot
        R = np.array(O[-1][-1 - i - 1])
        limite_inf, limite_sup = limites[-1 - i]
        if prismatica[-1 - i] == 1:
            w = 0
            w = sum(th[: i + 1])
            d = np.dot(np.array([cos(w), sin(w)]), objetivo - E)
            a[-1 - i] = max(limite_inf, min(limite_sup, a[-1 - i] + d))
        else:
            # Definir dos vectores
            v1 = E - R
            v2 = objetivo - R

            # Normalizar los vectores
            if np.linalg.norm(v1) != 0:
                v1 = v1 / np.linalg.norm(v1)
            else:
                v1 = np.zeros_like(v1)
            if np.linalg.norm(v2) != 0:
                v2 = v2 / np.linalg.norm(v2)
            else:
                v2 = np.zeros_like(v2)

            cos_alpha = np.dot(v1, v2)  # Producto punto (escalar)
            sin_alpha = np.cross(v1, v2)  # Producto cruzado para el determinante
            alpha = atan2(sin_alpha, cos_alpha)

            # Normalizar el ángulo a [-pi, pi]
            th[-1 - i] = (th[-1 - i] + alpha + np.pi) % (2 * np.pi) - np.pi

            # Ajustar el ángulo al rango permitido
            th[-1 - i] = max(limite_inf, min(limite_sup, th[-1 - i]))

        O.append(cin_dir(th, a))
    dist = np.linalg.norm(np.subtract(objetivo, O[-1][-1]))
    print("\n- Iteracion " + str(iteracion) + ":")
    muestra_origenes(O[-1])
    muestra_robot(O, objetivo, interactive=interactive)
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
