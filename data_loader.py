#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для загрузки данных плазмы.
"""

import os

os.add_dll_directory(r"C:\Program Files\netCDF 4.9.3\bin")
os.add_dll_directory(r"C:\Program Files\HDF_Group\HDF5\1.14.6\bin")

from boutdata import collect
import numpy as np
from constant import Nx, Tx, Vx, tp


def load_plasma_data(data_dir):
    """Загружает данные о плотности, температурах и скорости из указанной папки."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Директория {data_dir} не существует")

    original_dir = os.getcwd()
    os.chdir(data_dir)
    print(f"Рабочая директория изменена на: {os.getcwd()}")

    try:
        files = [f for f in os.listdir('.') if f.startswith('BOUT.dmp.') and f.endswith('.nc')]
        if not files:
            raise FileNotFoundError(f"В директории {data_dir} не найдены файлы BOUT.dmp.*.nc")
        print(f"Найденные файлы: {files}")

        n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0] * Nx
        Te = collect('Te', yguards=True)[:, 2:-2, 2:-2, 0] * Tx
        Ti = collect('Ti', yguards=True)[:, 2:-2, 2:-2, 0] * Tx
        Vz = collect('Vz', yguards=True)[:, 2:-2, 2:-2, 0] * Vx
        time = np.arange(0., n.shape[0] / 50 * tp, tp / 50) * 1e6  # time in microseconds
    finally:
        os.chdir(original_dir)
        print(f"Рабочая директория восстановлена: {os.getcwd()}")
    return n, Te, Ti, Vz, time