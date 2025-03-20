#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для моделирования магнитного поля.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from constant import Lcell, cell_L

def plot_magnetic_field():
    """Строит график магнитного поля Bz."""
    z1 = np.linspace(0, Lcell, 1000)
    z2 = np.linspace(Lcell, 35, 500)
    z3 = np.linspace(35, 40, 200)
    jz1 = np.exp(-((z1 - Lcell/2) / (Lcell/10))**2) * 10
    mu_0 = 4 * np.pi * 1e-7 * 1e4 * 10
    Bz1 = cumulative_trapezoid(jz1, z1, initial=0)
    Bz1 = Bz1 / np.max(Bz1) * 3
    B_end = Bz1[-1]
    Bz2 = (B_end - 1.3) * np.exp(-((z2 - Lcell) / ((35 - Lcell)/3))**2) + 1.3
    Bz3 = np.full_like(z3, 1.3)
    z = np.concatenate([z1, z2, z3])
    Bz = np.concatenate([Bz1, Bz2, Bz3])
    plt.figure(figsize=(14, 10))
    plt.plot(z, Bz, label=r'$B_z(z)$', color='blue', linewidth=2)
    plt.axvline(x=cell_L, color='green', linestyle='--', label='Начало накопительной ячейки, 7.1 см')
    plt.axvline(x=Lcell, color='red', linestyle='--', label='Конец накопительной ячейки, 28.6 см')
    plt.axvline(x=35, color='orange', linestyle='--', label='Начало постоянного поля, 35 см')
    plt.axhline(y=1.3, color='purple', linestyle=':', label='B(z) = 1.3 кГс')
    plt.xlabel('z, см', fontsize=28)
    plt.ylabel('Магнитное поле $B_z$, кГс', fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.ylim(0)
    plt.xlim(0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=20)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

# Закомментированная простая версия
def plot_simple_magnetic_field():
    z = np.linspace(0, Lcell, 1000)
    jz = np.exp(-((z - Lcell/2) / (Lcell/10))**2) * 10
    mu_0 = 4 * np.pi * 1e-7 * 1e4 * 10
    Bz = mu_0 * cumulative_trapezoid(jz, z, initial=0)
    plt.figure(figsize=(10, 6))
    plt.plot(z, Bz, label=r'$B_z(z)$', color='blue', linewidth=2)
    plt.axvline(x=7.1, color='green', linestyle='--', label='Начало накопительной ячейки (7.1 см)')
    plt.axvline(x=Lcell, color='red', linestyle='--', label='Конец накопительной ячейки (286 см)')
    plt.title('Распределение магнитного поля $B_z$ вдоль оси z', fontsize=14, fontweight='bold')
    plt.xlabel('Координата z, см', fontsize=12)
    plt.ylabel('Магнитное поле $B_z$, кГс', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()