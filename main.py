#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный модуль для запуска анализа плазмы.
"""
import os

from data_loader import load_plasma_data
from density_plots import (plot_density_by_radius, plot_density_2d, plot_density,
                          plot_avg_density, plot_density_by_diameter)
from temperature_plots import (plot_temperature_by_radius, plot_temperature_section,
                              plot_temperature_cross_section_fixed_z,
                              plot_temperature_sections_from_file)
from batch_processing import process_directories, process_directories_density_by_radius
from VAX import plot_iv_curve, plot_iv_curve_synthetic, plot_iv_curve_with_reconstruction
from magnetic_field import plot_magnetic_field, plot_simple_magnetic_field
from constant import BASE_PATH

def main():
    # Пример пути к одной папке с данными
    data_dir = os.path.join(BASE_PATH, 'Test1')

    # Загрузка данных из конкретной папки
    n, Te, Ti, Vz, time = load_plasma_data(data_dir)

    # Вызовы функций с указанием пути к данным
    plot_density_by_radius(nframes=50, z_positions=[7.0, 14.0], data_dir=data_dir, n_factor=4.2e16)
    plot_density_2d(np.log10(n), nframes=(12, 25, 50, 200), time=time, minvar=15, maxvar=19.5)
    plot_temperature_by_radius(nframes=5, z_positions=[30, 40], data_dir=data_dir, temp_type='Te', temp_factor=3.0e-2)
    plot_temperature_section(nframes=(12, 25, 50, 200), data_dir=data_dir, temp_type='Ti', temp_factor=3.0e-2)
    process_directories()  # уже работает с несколькими папками
    plot_iv_curve(data_dir)
    plot_magnetic_field()  # не требует данных

    # Закомментированные вызовы
    # plot_density(data_dir, (13, 25, 50, 200), 0, 1.25, 16., 10.)
    # plot_avg_density(data_dir, 16.5, 18.5, 8., 10.)
    # plot_density_by_diameter(nframes=50, z_positions=[7.0, 14.0], data_dir=data_dir, n_factor=4.2e16)
    # plot_temperature_cross_section_fixed_z(nframes=50, z_pos=7.0, data_dir=data_dir, num_theta=100, temp_type='Te', temp_factor=3.0e-2)
    # plot_temperature_sections_from_file(nframes=50, z_positions=[3.5, 7.0, 14.0, 21, 28.0], data_dir=data_dir, temp_type='Ti', temp_factor=3.0e-2)
    # plot_iv_curve_synthetic()
    # plot_iv_curve_with_reconstruction()
    # plot_simple_magnetic_field()
    # process_directories_density_by_radius()

if __name__ == "__main__":
    main()