#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для загрузки данных плазмы.
"""

import os
os.add_dll_directory(r"C:\Program Files\netCDF 4.9.2\bin")
os.add_dll_directory(r"C:\Program Files\HDF_Group\HDF5\1.14.6\bin")

from boutdata import collect
import numpy as np
from constant import Nx, Tx, Vx, tp


def load_plasma_data():
    """Загружает данные о плотности, температурах и скорости из текущей директории."""
    print(f"Рабочая директория: {os.getcwd()}")

    # Проверяем наличие файлов
    files = [f for f in os.listdir('.') if f.startswith('BOUT.dmp.') and f.endswith('.nc')]
    if not files:
        raise FileNotFoundError(f"В текущей директории {os.getcwd()} не найдены файлы BOUT.dmp.*.nc")
    print(f"Найденные файлы: {files}")

    n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0] * Nx
    Te = collect('Te', yguards=True)[:, 2:-2, 2:-2, 0] * Tx
    Ti = collect('Ti', yguards=True)[:, 2:-2, 2:-2, 0] * Tx
    Vz = collect('Vz', yguards=True)[:, 2:-2, 2:-2, 0] * Vx
    time = np.arange(0., n.shape[0] / 50 * tp, tp / 50) * 1e6  # time in microseconds
    return n, Te, Ti, Vz, time