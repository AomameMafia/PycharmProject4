#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для загрузки данных плазмы.
"""

from boutdata import collect
import numpy as np
from constant import Nx, Tx, Vx, tp

def load_plasma_data():
    """Загружает данные о плотности, температурах и скорости."""
    n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0] * Nx
    Te = collect('Te', yguards=True)[:, 2:-2, 2:-2, 0] * Tx
    Ti = collect('Ti', yguards=True)[:, 2:-2, 2:-2, 0] * Tx
    Vz = collect('Vz', yguards=True)[:, 2:-2, 2:-2, 0] * Vx
    time = np.arange(0., n.shape[0]/50*tp, tp/50) * 1e6  # time in microseconds
    return n, Te, Ti, Vz, time