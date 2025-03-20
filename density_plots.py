#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для построения графиков плотности плазмы.
"""
from datetime import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, ticker
from boutdata import collect
from constant import Rcell, Lcell, cell_R, cell_r, cell_L, Nx

def plot_density_by_radius(nframes, z_positions, n_factor=1.0):
    """График плотности по радиусу на заданных z."""
    n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]
    n_frame = n[nframes] * n_factor
    nx, ny = n_frame.shape
    r = np.linspace(0, Rcell, nx)
    z = np.linspace(0, Lcell, ny)
    plt.figure(figsize=(10, 6))
    for z_pos in z_positions:
        z_index = np.argmin(np.abs(z - z_pos))
        plt.plot(r, n_frame[:, z_index], label=f'z = {z_pos:.1f} см')
    plt.axvline(x=cell_R, color='red', linestyle='--', label='Граница накопительной ячейки')
    plt.xlabel('Радиус r, мм', fontsize=14, fontweight='bold')
    plt.ylabel('Плотность плазмы, × $10^{13}$ част/м³', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_density_2d(n, nframes, time, minvar, maxvar, figsizex=18.0, figsizey=8.0):
    """2D-график плотности в разные моменты времени."""
    nt, nx, ny = n.shape
    nf = len(nframes)
    x = np.linspace(0., Rcell, nx)
    y = np.linspace(0., Lcell, ny)
    Y, X = np.meshgrid(y, x)
    f = plt.figure(figsize=(figsizex / 2.54, figsizey / 2.54))
    wr = np.ones(nf)
    gs = gridspec.GridSpec(1, nf, width_ratios=wr)
    colorlev = np.linspace(minvar, maxvar, 100)
    cmap = plt.cm.hot_r
    for i in range(nf):
        idt = nframes[i]
        ax = f.add_subplot(gs[0, i])
        if i == 0:
            ax.set_ylabel(r'$z,\ \mathrm{cm}$', fontsize=28)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='y', which='both', bottom=False, left=False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))
        cs = ax.contourf(X, Y, n[idt, :, :], levels=colorlev, cmap=cmap, extend='both')
        ax.plot([cell_r, cell_R], [cell_L, cell_L], 'b', linewidth=2.0)
        ax.plot([cell_R, cell_R], [cell_L, Lcell], 'b', linewidth=2.0)
        ax.set_title('%d' % int(round(time[nframes[i]])), fontsize=20, pad=20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    cbar_ax = f.add_axes([0.9, 0.2, 0.015, 0.675])
    cax = f.colorbar(cs, cax=cbar_ax, ticks=np.linspace(minvar, maxvar, 5), orientation='vertical', extend='both')
    cax.ax.set_yticklabels(['%d' % t for t in np.linspace(minvar, maxvar, 5)])
    cax.ax.tick_params(labelsize=10)
    f.text(0.5, 0.04, r'$r,\ \mathrm{cm}$', ha='center', fontsize=28)
    f.suptitle(r'$t,\ \mathrm{\mu{s}}$:', fontsize=24, x=0.05, y=0.98)
    f.subplots_adjust(wspace=0.3, left=0.1, bottom=0.15, right=0.875, top=0.9)
    plt.show()

# Закомментированные функции
def plot_density(n, nframes, minvar, maxvar, figsizex=18.0, figsizey=8.0):
    nt, nx, ny = n.shape
    nf = len(nframes)
    x = np.linspace(0., Rcell, nx)
    y = np.linspace(0., Lcell, ny)
    Y, X = np.meshgrid(y, x)
    f = plt.figure(figsize=(figsizex/2.54, figsizey/2.54))
    wr = np.ones(nf)
    gs = gridspec.GridSpec(1, nf, width_ratios=wr)
    colorlev = np.linspace(minvar, maxvar, 100)
    cmap = plt.cm.hot_r
    for i in range(nf):
        idt = nframes[i]
        ax = f.add_subplot(gs[0, i])
        if i == 0:
            ax.set_ylabel(r'$z,\ \mathrm{cm}$', fontsize=28)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='y', which='both', bottom=False, left=False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))
        cs = ax.contourf(X, Y, n[idt, :, :], levels=colorlev, cmap=cmap, extend='both')
        ax.plot([cell_r, cell_R], [cell_L, cell_L], 'b', linewidth=2.0)
        ax.plot([cell_R, cell_R], [cell_L, Lcell], 'b', linewidth=2.0)
        ax.set_xlabel(r'$r,\ \mathrm{cm}$', fontsize=28)
        ax.set_title('$t\ =\ {%s}\ \mathrm{\mu{s}}$' % str(round(time[nframes[i]], 2)), fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    cbar_ax = f.add_axes([0.9, 0.2, 0.015, 0.675])
    cax = f.colorbar(cs, cax=cbar_ax, ticks=np.linspace(minvar, maxvar, 5), orientation='vertical', extend='both')
    cax.ax.tick_params(labelsize=10)
    f.subplots_adjust(wspace=0.0, left=0.1, bottom=0.1, right=0.875, top=0.95)
    plt.show()

def plot_avg_density(n, minvar, maxvar, figsizex=18.0, figsizey=8.0):
    nt, nx, ny = n.shape
    x = np.linspace(0., 1., nx)
    y = np.linspace(0., 1., ny)
    Y, X = np.meshgrid(y, x)
    f = plt.figure(figsize=(figsizex/2.54, figsizey/2.54))
    plt.xlabel(r'$r/R$', fontsize=12)
    plt.ylabel(r'$z/L$', fontsize=12)
    colorlev = np.linspace(minvar, maxvar, 100)
    cmap = plt.cm.hot_r
    cs = plt.contourf(X, Y, np.mean(n[10:,:,:], axis=0), levels=colorlev, cmap=cmap, extend='both')
    ax = f.gca()
    ax.xaxis.set_major_locator(ticker.FixedLocator([0., 0.25, 0.5, 0.75]))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(ticker.FixedLocator([0., 0.25, 0.5, 0.75, 1.0]))
    cbar_ax = f.add_axes([0.85, 0.13, 0.015, 0.82])
    cax = f.colorbar(cs, cax=cbar_ax, ticks=np.arange(minvar, maxvar, 0.5), orientation='vertical', extend='both')
    cax.ax.tick_params(labelsize=10)
    f.subplots_adjust(wspace=0.0, left=0.19, bottom=0.13, right=0.8, top=0.95)
    plt.show()

def plot_density_by_diameter(nframes, z_positions, n_factor=1.0):
    n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]
    n_140us = n[nframes] * n_factor
    nx, ny = n_140us.shape
    d = np.linspace(-Rcell, Rcell, 2 * nx)
    z = np.linspace(0, Lcell, ny)
    plt.figure(figsize=(12, 8))
    for z_pos in z_positions:
        z_index = np.argmin(np.abs(z - z_pos))
        profile = np.concatenate([n_140us[::-1, z_index], n_140us[:, z_index]])
        plt.plot(d, profile, label=f'z = {z_pos:.1f} см')
    plt.axvline(x=-cell_R, color='red', linestyle='--', label='Левая граница накопительной ячейки')
    plt.axvline(x=cell_R, color='red', linestyle='--', label='Правая граница накопительной ячейки')
    plt.title('Распределение плотности плазмы по диаметру', fontsize=18, fontweight='bold')
    plt.xlabel('Диаметр d, см', fontsize=16, fontweight='bold')
    plt.ylabel('Плотность плазмы, × $10^{13}$ част/м³', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()