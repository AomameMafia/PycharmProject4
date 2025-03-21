#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для массовой обработки данных из нескольких папок.
"""

import os
os.add_dll_directory(r"C:\Program Files\netCDF 4.9.3\bin")
os.add_dll_directory(r"C:\Program Files\HDF_Group\HDF5\1.14.6\bin")
import matplotlib.pyplot as plt
import numpy as np
from boutdata import collect
from constant import Lcell, cell_L, Rcell, cell_R, BASE_PATH

def plot_density_z_for_directory(data_dir, nframes):
    """График плотности по оси z для одной папки."""
    os.chdir(data_dir)
    n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]
    max_index = n.shape[0] - 1
    if nframes > max_index:
        print(f"Предупреждение: nframes={nframes} превышает доступный размер {max_index + 1}. Используется последний кадр.")
        nframes = max_index
    n_140us = n[nframes]
    n_z_profile = np.mean(n_140us, axis=0) * 4.2e16
    z = np.linspace(0, Lcell, n_z_profile.shape[0])
    plt.figure(figsize=(16, 12))
    plt.plot(z, n_z_profile, label=f'Плотность плазмы n(z), папка: {os.path.basename(data_dir)}', linewidth=2)
    plt.axvline(x=cell_L, color='blue', linestyle='--', label='Начало накопительной ячейки 7.1 см')
    plt.title(f'График плотности для {os.path.basename(data_dir)}', fontsize=28, fontweight='bold')
    plt.xlabel('z, см', fontsize=28, fontweight='bold')
    plt.ylabel(r'Плотность плазмы × $10^{13}$ част/м$^3$', fontsize=28, fontweight='bold')
    plt.xlim(0, Lcell)
    plt.ylim(0)
    plt.xticks(np.arange(0, Lcell + 1, 2))
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

def process_directories():
    """Обрабатывает все папки Test1-Test62."""
    data_dirs = [os.path.join(BASE_PATH, f'Test{i}') for i in range(1, 63)]
    failed_dirs = []
    for data_dir in data_dirs:
        folder_name = os.path.basename(data_dir)
        try:
            plot_density_z_for_directory(data_dir, nframes=50)
        except (OSError, FileNotFoundError, IndexError) as e:
            print(f"Ошибка с папкой {folder_name}: {e}")
            failed_dirs.append(folder_name)
    if failed_dirs:
        print("\nПроблемы возникли с папками:")
        for folder in failed_dirs:
            print(f"- {folder}")
    else:
        print("\nВсе папки обработаны успешно!")

def plot_density_by_radius_multi(data_dir, title, nframes, z_positions, n_factor=1.0):
    os.chdir(data_dir)
    n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]
    max_index = n.shape[0] - 1
    if nframes > max_index:
        print(f"Предупреждение: nframes={nframes} превышает доступный размер {max_index + 1}. Используется последний кадр.")
        nframes = max_index
    n_140us = n[nframes] * n_factor
    nx, ny = n_140us.shape
    r = np.linspace(0, Rcell, nx)
    z = np.linspace(0, Lcell, ny)
    plt.figure(figsize=(10, 6))
    for z_pos in z_positions:
        z_index = np.argmin(np.abs(z - z_pos))
        plt.plot(r, n_140us[:, z_index], label=f'z = {z_pos:.1f} см')
    plt.axvline(x=cell_R, color='red', linestyle='--', label='Граница накопительной ячейки')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Радиус r, мм', fontsize=14, fontweight='bold')
    plt.ylabel('Плотность плазмы, × $10^{19}$ част/м³', fontsize=14, fontweight='bold')
    plt.xlim(0, Rcell)
    plt.legend(fontsize=16)
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.show()

def process_directories_density_by_radius():
    data_dirs = [os.path.join(BASE_PATH, f'Test{i}') for i in range(1, 63)]
    failed_dirs = []
    for data_dir in data_dirs:
        folder_name = os.path.basename(data_dir)
        try:
            plot_density_by_radius_multi(data_dir, f'График для {folder_name}', nframes=100, z_positions=[1.0, 2.0, 3.0], n_factor=1.0)
        except (OSError, FileNotFoundError, IndexError) as e:
            print(f"Ошибка с папкой {folder_name}: {e}")
            failed_dirs.append(folder_name)
    if failed_dirs:
        print("\nПроблемы возникли с папками:")
        for folder in failed_dirs:
            print(f"- {folder}")
    else:
        print("\nВсе папки обработаны успешно!")