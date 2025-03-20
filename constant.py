#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль с константами для моделирования плазмы.
"""

import numpy as np

# Физические константы
A = 1.0
tau_p = 200e-6  # pulse duration [s]
t_plot = 100
y_plot = 60
nt_tot = 200
Lcell = 4.5e-1  # plasma cell length [m]
Rcell = 3.6e-2  # plasma cell radius [m]
Nx = 4.2e16     # residual particle density [1/m^3]
Tx = 3.0e-2     # residual plasma temperature [eV]
tp = 140e-6     # pulse duration [s]
extractor_r = 9e-3  # m
cell_R = 9.0e-3  # m - radius of the plasma cell
cell_L = Lcell / 4.0  # m - length of the plasma cell
cell_r = 2.5e-3  # m - radius of the cell hole

# Преобразование в см
Rcell *= 1e2
extractor_r *= 1e2
cell_R *= 1e2
cell_L *= 1e2
cell_r *= 1e2

# Вспомогательные переменные
Vx = np.sqrt(Tx * 1.6e-19 / A / 1.67e-27)  # plasma exit velocity [m/s]

# Константы для ВАХ
e = 1.6e-19      # заряд электрона, Кл
me = 9.11e-31    # масса электрона, кг
kB = 1.38e-23    # постоянная Больцмана, Дж/К
probe_area = 6e-5  # площадь зонда, м²
amplitude = 50    # амплитуда напряжения, В
num_points = 100  # число точек развертки

# Дополнительные параметры для ВАХ (из закомментированных участков)
sampling_rate = 1000  # частота выборки
impulse_duration = 140e-6  # длительность импульса, с