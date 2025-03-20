#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для расчета и построения ВАХ.
"""

import numpy as np
import matplotlib.pyplot as plt
from constant import e, me, kB, probe_area, amplitude, num_points, sampling_rate, impulse_duration
from data_loader import load_plasma_data

def current_density(V, n_e, Te_e):
    """Рассчитывает плотность тока."""
    v_th = np.sqrt(kB * Te_e * e / me)
    exponent = np.clip(-e * V / (kB * Te_e * e), -50, 50)
    return e * n_e * v_th * (1 - np.exp(exponent))

def plot_iv_curve():
    """Строит ВАХ на основе данных."""
    n, Te, _, _, _ = load_plasma_data()
    V_probe = np.linspace(-amplitude, amplitude, num_points)
    n = np.mean(n, axis=(1, 2))
    Te = np.mean(Te, axis=(1, 2))
    n = np.interp(np.linspace(0, len(n)-1, num_points), np.arange(len(n)), n)
    Te = np.interp(np.linspace(0, len(Te)-1, num_points), np.arange(len(Te)), Te)
    I_probe = np.array([current_density(V, n[i], Te[i]) * probe_area for i, V in enumerate(V_probe)])
    plt.figure(figsize=(10, 6))
    plt.plot(V_probe, I_probe, label='ВАХ')
    plt.title('Вольт-амперная характеристика (ВАХ)', fontsize=16)
    plt.xlabel('Напряжение, В', fontsize=14)
    plt.ylabel('Ток, А', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

# Закомментированные варианты ВАХ
def plot_iv_curve_synthetic():
    V_probe = np.linspace(-amplitude, amplitude, num_points)
    n = np.linspace(1e16, 1e17, num_points)
    Te = np.linspace(2, 5, num_points)
    I_probe = np.array([current_density(V, n[i], Te[i]) * probe_area for i, V in enumerate(V_probe)])
    mask_neg = V_probe < -10
    I_sat = np.mean(I_probe[mask_neg])
    mask_exp = (V_probe > 5) & (I_probe > I_sat * 1.01)
    V_exp = V_probe[mask_exp]
    I_exp = I_probe[mask_exp]
    log_I = np.log(I_exp - I_sat)
    coeffs = np.polyfit(V_exp, log_I, 1)
    slope = coeffs[0]
    Te_reconstructed = -e / (kB * slope)
    print(f"Восстановленная температура электронов: {Te_reconstructed:.2f} эВ")
    plt.figure(figsize=(10, 6))
    plt.plot(V_probe, I_probe, label='ВАХ', color='blue')
    plt.axhline(I_sat, color='red', linestyle='--', label=f'I_sat = {I_sat:.2e} A')
    plt.plot(V_exp, I_exp, 'go', label='Экспоненциальный участок')
    plt.title('Вольт-Амперная характеристика (ВАХ)', fontsize=16)
    plt.xlabel('Напряжение, В', fontsize=14)
    plt.ylabel('Ток, А', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(V_exp, log_I, 'o', label='ln(I - I_sat)')
    plt.plot(V_exp, coeffs[0] * V_exp + coeffs[1], label=f'Аппроксимация (slopeThen slope={slope:.3f})', color='orange')
    plt.title('Логарифмический график тока на экспоненциальном участке', fontsize=16)
    plt.xlabel('Напряжение, В', fontsize=14)
    plt.ylabel('ln(I - I_sat)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_iv_curve_with_reconstruction():
    num_points = max(1, int(impulse_duration * sampling_rate))
    time = np.linspace(0, impulse_duration, num_points)
    V_probe = np.linspace(-amplitude, amplitude, num_points)
    base_density = 1e18
    base_temperature = 5
    n_avg = base_density * (1 + 0.1 * np.sin(2 * np.pi * time / impulse_duration))
    Te_avg = base_temperature * (1 + 0.15 * np.cos(2 * np.pi * time / impulse_duration))
    n_e_dynamic = np.random.normal(1, 0.1, len(time)) * n_avg
    Te_e_dynamic = np.random.normal(1, 0.15, len(time)) * Te_avg
    I_probe_dynamic = np.zeros_like(V_probe)
    for i in range(num_points):
        I_probe_dynamic[i] = current_density(V_probe[i], n_e_dynamic[i], Te_e_dynamic[i]) * probe_area
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(time * 1e6, n_e_dynamic, label='Концентрация электронов')
    plt.title('Динамика концентрации')
    plt.xlabel('Время, мкс')
    plt.ylabel('Концентрация, м^-3')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(time * 1e6, Te_e_dynamic, label='Температура электронов')
    plt.title('Динамика температуры')
    plt.xlabel('Время, мкс')
    plt.ylabel('Температура, эВ')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(V_probe, I_probe_dynamic)
    plt.title('ВАХ с динамическими параметрами')
    plt.xlabel('Напряжение, В')
    plt.ylabel('Ток, А')
    plt.tight_layout()
    plt.show()

def reconstruct_parameters(V, I):
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    if V.size == 0 or I.size == 0:
        raise ValueError("Input arrays must not be empty")
    if V.size != I.size:
        raise ValueError("Input arrays must have the same length")
    neg_voltage_mask = V < 0
    if np.any(neg_voltage_mask):
        I_sat_neg = np.max(I[neg_voltage_mask])
    else:
        I_sat_neg = np.max(I)
    try:
        valid_mask = (I > 0) & (np.abs(V) > 0)
        log_current = np.log(np.abs(I[valid_mask]) + 1e-10)
        voltage_subset = V[valid_mask]
        if len(voltage_subset) < 2:
            raise ValueError("Insufficient valid data points")
        coeffs = np.polyfit(voltage_subset, log_current, 1)
        e = 1.6e-19
        T_e_slope = -1 / coeffs[0]
        T_e_reconstructed = T_e_slope * e / kB
        T_e_reconstructed = max(0.1, min(T_e_reconstructed, 100))
        n_e_reconstructed = I_sat_neg / (e * probe_area * np.sqrt(2 * np.pi * kB * T_e_reconstructed / me))
        n_e_reconstructed = max(1e16, min(n_e_reconstructed, 1e20))
        return n_e_reconstructed, T_e_reconstructed
    except Exception as e:
        print(f"Error in parameter reconstruction: {e}")
        return 1e18, 5.0