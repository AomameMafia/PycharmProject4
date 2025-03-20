#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный модуль для запуска анализа плазмы.
"""
import numpy as np

from data_loader import load_plasma_data
from density_plots import plot_density_by_radius, plot_density_2d
from temperature_plots import plot_temperature_by_radius, plot_temperature_section
from batch_processing import process_directories
from VAX import plot_iv_curve
from magnetic_field import plot_magnetic_field

def main():
    # Загрузка данных
    n, Te, Ti, Vz, time = load_plasma_data()

    # Примеры вызовов функций
    plot_density_by_radius(nframes=50, z_positions=[7.0, 14.0], n_factor=4.2e16)
    plot_density_2d(np.log10(n), nframes=(12, 25, 50, 200), time=time, minvar=15, maxvar=19.5)
    plot_temperature_by_radius(nframes=5, z_positions=[30, 40], temp_type='Te', temp_factor=3.0e-2)
    plot_temperature_section(nframes=(12, 25, 50, 200), temp_type='Ti', temp_factor=3.0e-2)
    process_directories()
    plot_iv_curve()
    plot_magnetic_field()

if __name__ == "__main__":
    main()