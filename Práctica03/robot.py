#! /usr/bin/env python
# -*- coding: utf-8 -*-


from math import *
import random
import numpy as np
import copy


class robot:
    def __init__(self):
        # Inicializacion de pose y parámetros de ruido
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        self.weight = 1.0
        self.old_weight = 1.0
        self.size = 1.0

    def copy(self):
        # Constructor de copia
        return copy.deepcopy(self)

    def set(self, new_x, new_y, new_orientation):
        # Modificar la pose
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
        # Normalizar ángulo orientación
        while self.orientation > pi:
            self.orientation -= 2 * pi
        while self.orientation < -pi:
            self.orientation += 2 * pi

    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # Modificar los parámetros de ruido
        self.forward_noise = float(new_f_noise)
        self.turn_noise = float(new_t_noise)
        self.sense_noise = float(new_s_noise)

    def pose(self):
        # Obtener pose actual
        return [self.x, self.y, self.orientation]

    def sense1(self, landmark, noise):
        # Calcular la distancia a una de las balizas
        return np.linalg.norm(np.subtract([self.x, self.y], landmark)) + random.gauss(
            0.0, noise
        )

    def sense(self, landmarks):
        # Calcular las distancias a cada una de las balizas
        d = [self.sense1(l, self.sense_noise) for l in landmarks]
        # En la última posición se añade la orientación
        d.append(self.orientation + random.gauss(0.0, self.sense_noise))
        return d

    def move(self, turn, forward):
        # Modificar pose del robot (holonómico) - NO necesita avanzar para girar
        self.orientation += float(turn) + random.gauss(0.0, self.turn_noise)
        # Normalizar ángulo orientación
        while self.orientation > pi:
            self.orientation -= 2 * pi
        while self.orientation < -pi:
            self.orientation += 2 * pi
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        self.x += cos(self.orientation) * dist
        self.y += sin(self.orientation) * dist

    def move_triciclo(self, turn, forward, largo):
        # Modificar pose del robot (Ackermann) - Necesita avanzar para girar
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        self.orientation += dist * tan(float(turn)) / largo + random.gauss(
            0.0, self.turn_noise
        )
        while self.orientation > pi:
            self.orientation -= 2 * pi
        while self.orientation < -pi:
            self.orientation += 2 * pi
        self.x += cos(self.orientation) * dist
        self.y += sin(self.orientation) * dist

    def Gaussian(self, mu, sigma, x):
        # Calcular la probabilidad de 'x' para una distribución normal
        # de media 'mu' y desviación típica 'sigma'
        if sigma:
            return exp(-(((mu - x) / sigma) ** 2) / 2) / (sigma * sqrt(2 * pi))
        else:
            return 0

    def measurement_prob(self, measurements, landmarks):
        # Dado que la pose real del robot no es directamente accesible, se utilizan las
        # mediciones de las balizas (landmarks) para evaluar qué tan cerca está la estimación
        # del robot ideal de las medidas obtenidas (del robot real).
        #
        # Este método calcula el error promedio entre:
        # - Las distancias estimadas (por el robot ideal) a cada baliza y las distancias medidas (del robot real).
        # - La orientación estimada y la orientación medida.
        #
        # Un valor de error (self.weight) más alto indica mayor discrepancia entre las
        # mediciones reales y las estimaciones del robot ideal, es decir, menor confiabilidad.
        # Por el contrario, un valor bajo de self.weight indica mayor probabilidad de que
        # la posición y orientación estimadas sean correctas.

        self.weight = 0.0
        n = 0
        # Acumula las diferencias absolutas entre las distancias estimadas y las medidas reales para cada baliza.
        for i in range(len(measurements) - 1):
            self.weight += abs(self.sense1(landmarks[i], 0) - measurements[i])
            n += 1

        # Calcula la diferencia en la orientación estimada frente a la orientación medida.
        diff = self.orientation - measurements[-1]

        # Normaliza el ángulo resultante al rango [-pi, pi].
        while diff > pi:
            diff -= 2 * pi
        while diff < -pi:
            diff += 2 * pi

        # Suma la diferencia de orientación al error total.
        self.weight += abs(diff)

        # Calcula el promedio del error total, considerando las distancias a las balizas y la orientación.
        self.weight /= n + 1

        return self.weight

    def __repr__(self):
        # Representación de la clase robot
        return "[x=%.6s y=%.6s orient=%.6s]" % (
            str(self.x),
            str(self.y),
            str(self.orientation),
        )
