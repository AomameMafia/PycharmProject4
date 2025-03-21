#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для построения графиков температуры плазмы.
"""

import os
os.add_dll_directory(r"C:\Program Files\netCDF 4.9.3\bin")
os.add_dll_directory(r"C:\Program Files\HDF_Group\HDF5\1.14.6\bin")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from boutdata import collect
from constant import Rcell, Lcell, cell_R

def plot_temperature_by_radius(nframes, z_positions, data_dir=None, temp_type='Te', temp_factor=1.0):
    """График температуры по радиусу на заданных z."""
    if data_dir:
        original_dir = os.getcwd()
        os.chdir(data_dir)
    try:
        temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]
        temp_140us = temp_data[nframes] * temp_factor
        nx, ny = temp_140us.shape
        r = np.linspace(0, Rcell, nx)
        z = np.linspace(0, Lcell, ny)
        plt.figure(figsize=(10, 6))
        for z_pos in z_positions:
            z_index = np.argmin(np.abs(z - z_pos))
            plt.plot(r, temp_140us[:, z_index], label=f'z = {z_pos:.1f} см')
        plt.axvline(x=cell_R, color='red', linestyle='--', label='Граница накопительной ячейки')
        plt.xlabel('Радиус r, см', fontsize=28, fontweight='bold')
        plt.ylabel('Температура, эВ', fontsize=28, fontweight='bold')
        plt.xlim(0, Rcell)
        plt.ylim(0)
        plt.grid(False)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.show()
    finally:
        if data_dir:
            os.chdir(original_dir)

def plot_temperature_section(nframes, data_dir=None, temp_type='Ti', temp_factor=1.0, figsizex=18.0, figsizey=8.0):
    """2D-график температуры в разные моменты времени."""
    if data_dir:
        original_dir = os.getcwd()
        os.chdir(data_dir)
    try:
        temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]
        num_frames = len(nframes)
        f = plt.figure(figsize=(figsizex / 2.54 * num_frames, figsizey / 2.54))
        gs = gridspec.GridSpec(1, num_frames, width_ratios=[1]*num_frames)
        nx, ny = temp_data[0, :, :].shape
        r = np.linspace(0, Rcell, nx)
        z = np.linspace(0, Lcell, ny)
        min_temp = (temp_data.min() * temp_factor)
        max_temp = (temp_data.max() * temp_factor)
        for idx, frame in enumerate(nframes):
            temp_frame = temp_data[frame] * temp_factor
            ax = f.add_subplot(gs[idx])
            contour = ax.contourf(r, z, temp_frame.T, levels=50, cmap='inferno', vmin=min_temp, vmax=max_temp)
            ax.plot([0, Rcell], [Lcell / 4, Lcell / 4], 'b', linewidth=2.0)
            ax.plot([Rcell, Rcell], [0, Lcell / 4], 'b', linewidth=2.0)
            ax.set_title(f'Время: {frame} мкс', fontsize=12, fontweight='bold')
            ax.set_xlabel('Радиус r, см', fontsize=10, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Координата z, см', fontsize=10, fontweight='bold')
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='major', labelsize=8)
        cbar = f.colorbar(contour, ax=f.get_axes(), orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(f'{temp_type}, эВ', fontsize=12, fontweight='bold')
        cbar.set_ticks([min_temp, (min_temp + max_temp) / 2, max_temp])
        cbar.set_ticklabels(['%d' % t for t in [min_temp, (min_temp + max_temp) / 2, max_temp]])
        plt.tight_layout()
        plt.show()
    finally:
        if data_dir:
            os.chdir(original_dir)

def plot_temperature_cross_section_fixed_z(nframes, z_pos, data_dir=None, num_theta=100, temp_type='Te', temp_factor=1.0):
    if data_dir:
        original_dir = os.getcwd()
        os.chdir(data_dir)
    try:
        temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]
        temp_140us = temp_data[nframes] * temp_factor
        nx, ny = temp_140us.shape
        r = np.linspace(0, Rcell, nx)
        z = np.linspace(0, Rcell, ny)
        z_index = np.argmin(np.abs(z - z_pos))
        temp_at_z = temp_140us[:, z_index]
        theta = np.linspace(0, 2 * np.pi, num_theta)
        r_grid, theta_grid = np.meshgrid(r, theta)
        temp_2d = np.tile(temp_at_z, (num_theta, 1))
        x = r_grid * np.cos(theta_grid)
        y = r_grid * np.sin(theta_grid)
        plt.figure(figsize=(8, 8))
        plt.contourf(x, y, temp_2d, levels=50, cmap='inferno')
        plt.colorbar(label=f'{temp_type}, эВ')
        plt.title(f'Сечение {temp_type} на z = {z_pos:.1f} см', fontsize=14, fontweight='bold')
        plt.xlabel('Координата x, см', fontsize=12, fontweight='bold')
        plt.ylabel('Координата y, см', fontsize=12, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    finally:
        if data_dir:
            os.chdir(original_dir)



#
# def plot_temperature_sections_from_file(nframes, z_positions, data_dir=None, temp_type='Ti', temp_factor=1.0):
#     if data_dir:
#         original_dir = os.getcwd()
#         os.chdir(data_dir)
#     try:
#         temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]
#         temp_140us = temp_data[nframes] * temp_factor
#         nx, ny = temp_140us.shape
#         r = np.linspace(0, Rcell, nx)
#         z = np.linspace(0, Lcell, ny)
#         n_plots = len(z_positions)
#         fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6), subplot_kw={'aspect': 'equal'})
#         if n_plots == 1:
#             axes = [axes]
#         for ax, z_pos in zip(axes, z_positions):
#             z_index = np.argmin(np.abs(z - z_pos))
#             temp_at_z = temp_140us[:, z_index]
#             theta = np.linspace(0, 2 * np.pi, nx)
#             r_grid, theta_grid = np.meshgrid(r, theta)
#             temp_2d = np.tile(temp_at_z, (nx, 1))
#             x = r_grid * np.cos(theta_grid)
#             y = r_grid * np.sin(theta_grid)
#             cs = ax.contourf(x, y, temp_2d, levels=20, cmap='inferno', extend='both')
#             ax.set_title(f'z = {z_pos:.1f} см', fontsize=14, fontweight='bold')
#             ax.set_xlabel('x, см', fontsize=16)
#             ax.set_ylabel('y, см', fontsize=16)
#             cbar = fig.colorbar(cs, ax=ax, orientation='vertical', label=f'{temp_type}, эВ')
#             cbar.ax.tick_params(labelsize=14)
#         plt.tight_layout()
#         plt.show()
#     finally:
#         if data_dir:
#             os.chdir(original_dir)



if __name__ == "__main__":
    try:
        # Пример вызова функций с параметрами
        plot_temperature_by_radius(50, [7.0, 14.0], temp_type='Te', temp_factor=1.0)
        plot_temperature_section([12, 25, 50], temp_type='Ti', temp_factor=1.0)
        plot_temperature_cross_section_fixed_z(50, 7.0, temp_type='Te', temp_factor=1.0)
        # plot_temperature_sections_from_file(50, [7.0, 14.0], temp_type='Ti', temp_factor=1.0)
    except Exception as e:
        print(f"Ошибка при построении графиков: {e}")