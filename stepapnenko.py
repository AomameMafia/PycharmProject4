#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:59:43 2024

@author: aegis
"""

from boutdata import collect
from boututils import showdata
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import trapezoid
import numpy as np
import matplotlib.pyplot as plt
from boutdata import collect

A = 1.0
tau_p = 200e-6  # pulse duration [s]
t_plot = 100
y_plot = 60
nt_tot = 200
Lcell = 4.5e-1  # plasma cell length [m]
Rcell = 3.6e-2  # plasma cell radius [m]
# Nx = 5e16     # residual particle density [1/m^3]
# Tx = 3e-2     # residual plasma temperature eVK]
# Vx = 5.0e4    # plasma exit velocity [m/s]
Nx = 4.2e16  # residual particle density [1/m^3]
Tx = 3.0e-2  # residual plasma temperature [eV]
tp = 140e-6  # pulse duration [s]
extractor_r = 9e-3  # m

cell_R = 9.0e-3  # m - radius of the plasma cell
cell_L = Lcell / 4.0  # m - length of the plasma cell
cell_r = 2.5e-3  # m - radius of the cell hole
# cell_R /= Rcell      # normalized
# cell_L = 1.0 - 0.5   # normalized coordinate
# cell_r /= Rcell      # normalized


Rcell *= 1e2
extractor_r *= 1e2
cell_R *= 1e2
cell_L *= 1e2
cell_r *= 1e2

# auxiliary variables
Vx = np.sqrt(Tx * 1.6e-19 / A / 1.67e-27)  # plasma exit velocity [m/s]

# os.chdir(r'C:\Users\Alisher\Desktop\fromcluster\Test21')

# C:\Users\Alisher\Desktop\fromcluster
#


# def plot_density_by_radius(nframes, z_positions, Rcell, Lcell, cell_R, n_factor=1.0):
#     """
#     Построить графики распределения плотности плазмы по радиусу на заданных расстояниях z.

#     Parameters:
#     - nframes: int, индекс временного кадра
#     - z_positions: list, координаты z (в см), на которых строится распределение
#     - Rcell: float, радиус ячейки в см
#     - Lcell: float, длина ячейки в см
#     - cell_R: float, граница накопительной ячейки в см
#     - n_factor: float, множитель для масштабирования плотности

#     Returns:
#     - None
#     """
#     # Загружаем плотность плазмы
#     n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0] # Из файла, уже настроено
#     n_140us = n[nframes] * n_factor # Берем кадр и масштабируем

#     # Координаты
#     nx, ny = n_140us.shape
#     r = np.linspace(0, Rcell, nx) # Радиальная координата
#     z = np.linspace(0, Lcell, ny) # Продольная координата

#     # Построение графиков
#     plt.figure(figsize=(10, 6))

#     # # Устанавливаем начало оси Y с нуля
#     # plt.ylim(bottom=0)

#     for z_pos in z_positions:
#         z_index = np.argmin(np.abs(z - z_pos)) # Ищем ближайший индекс
#         plt.plot(r, n_140us[:, z_index], label=f'z = {z_pos:.1f} см')

#     # Добавляем границу накопительной ячейки
#     plt.axvline(x=cell_R, color='red', linestyle='--', label='Граница накопительной ячейки')

#     # Настройка графика
#     plt.xlabel('Радиус r, мм', fontsize=14, fontweight='bold')
#     plt.ylabel('Плотность плазмы, × $10^{20}$ част/м³', fontsize=14, fontweight='bold')
#     plt.xlim(0, Rcell)  # Устанавливаем начало оси X с нуля
#     plt.legend(fontsize=16)

#     plt.grid(False)
#     plt.tick_params(axis='both', which='major', labelsize=14)
#     plt.tight_layout()
#     plt.show()


# # Пример вызова функции (для использования в вашем анализе):
# plot_density_by_radius(nframes=26, z_positions=[11,20], Rcell=36, Lcell=276, cell_R=9, n_factor=4.2e16)


# n = collect('n', yguards=True)[:,2:-2,2:-2,0] * Nx
# Te = collect('Te', yguards=True)[:,2:-2,2:-2,0] * Tx
# Ti = collect('Ti', yguards=True)[:,2:-2,2:-2,0] * Tx
# Vz = collect('Vz', yguards=True)[:,2:-2,2:-2,0] * Vx
# time = np.arange(0., n.shape[0]/50*tp, tp/50) * 1e6 # time in microseconds

# def plot_density(n, nframes, minvar, maxvar, figsizex = 18.0, figsizey = 8.0):
#     nt, nx, ny = n.shape
#     nf = len(nframes)

#     x = np.linspace(0., Rcell, nx)
#     y = np.linspace(0., Lcell, ny)
#     Y, X  = np.meshgrid(y, x)

#     f = plt.figure(figsize=(figsizex/2.54, figsizey/2.54)) # plot size is defined in cms
#     wr = np.ones(nf)
#     gs = gridspec.GridSpec(1, nf, width_ratios=wr)

#     colorlev = np.linspace(minvar, maxvar, 100)
#     cmap = plt.cm.hot_r

#     for i in range(nf):
#         idt = nframes[i]
#         ax = f.add_subplot(gs[0, i])
#         if i == 0:
#             ax.set_ylabel(r'$z,\ \mathrm{cm}$', fontsize=28)
#         else:
#             plt.setp(ax.get_yticklabels(), visible = False)
#             ax.tick_params(axis = 'y', which = 'both', bottom = False, left = False)

#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
#         ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#         ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

#         cs = ax.contourf(X, Y, n[idt, :, :],
#                          levels = colorlev,
#                          cmap = cmap,
#                          extend='both')
#         ax.plot([cell_r, cell_R], [cell_L, cell_L], 'b', linewidth=2.0)
#         ax.plot([cell_R, cell_R], [cell_L, Lcell], 'b', linewidth=2.0)
#         # ax.set_xlim([0., 0.5])
#         ax.set_xlabel(r'$r,\ \mathrm{cm}$', fontsize=28)
#         ax.set_title('$t\ =\ {%s}\ \mathrm{\mu{s}}$' % str(round(time[nframes[i]], 2)), fontsize = 12)

#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
#     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

#     cbar_ax = f.add_axes([0.9, 0.2, 0.015, 0.675])
#     cax = f.colorbar(cs, cax=cbar_ax, ticks=np.linspace(minvar, maxvar, 5), orientation = 'vertical', extend = 'both')

#     cax.ax.tick_params(labelsize=10)


#     f.subplots_adjust(wspace = 0.0, left=0.1, bottom=0.1, right=0.875, top=0.95)

#     plt.show()

# def plot_avg_density(n, minvar, maxvar, figsizex = 18.0, figsizey = 8.0):
#     nt, nx, ny = n.shape
#     nf = len(nframes)

#     x = np.linspace(0., 1., nx)
#     y = np.linspace(0., 1., ny)
#     Y, X  = np.meshgrid(y, x)

#     f = plt.figure(figsize=(figsizex/2.54,figsizey/2.54))
#     plt.xlabel(r'$r/R$', fontsize=12)
#     plt.ylabel(r'$z/L$', fontsize=12)

#     colorlev = np.linspace(minvar, maxvar, 100)
#     cmap = plt.cm.hot_r
#     cs = plt.contourf(X, Y, np.mean(n[10:,:,:], axis=0),
#                       levels = colorlev,
#                       cmap = cmap,
#                       extend='both')

#     ax = f.gca()
#     ax.xaxis.set_major_locator(ticker.FixedLocator([0., 0.25, 0.5, 0.75]))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
#     ax.xaxis.set_major_locator(ticker.FixedLocator([0., 0.25, 0.5, 0.75, 1.0]))

#     cbar_ax = f.add_axes([0.85, 0.13, 0.015, 0.82])
#     cax = f.colorbar(cs, cax=cbar_ax, ticks=np.arange(minvar, maxvar, 0.5), orientation = 'vertical', extend = 'both')
#     # cax = f.colorbar(cs, cax=cbar_ax, ticks=[16.5, 17.5, 18.5, 19.5, 20.5], orientation = 'vertical', extend = 'both')
#     # cax = f.colorbar(cs, cax=cbar_ax, ticks=[1e19, 2e19, 3e19, 4e19, 5e19], orientation = 'vertical', extend = 'both')
#     # cax = f.colorbar(cs, cax=cbar_ax, ticks=[0.2e20, 0.4e20, 0.6e20, 0.8e20, 1.0e20], orientation = 'vertical', extend = 'both')
#     # cax = f.colorbar(cs, cax=cbar_ax, ticks=[1e19, 2e19, 3e19, 4e19, 5e19], orientation = 'vertical', extend = 'max')
#     # cax.ax.set_yticklabels([round(minvar + i*(3.0-minvar)/3, 1) for i in range(0, 4)])  # vertical colorbar
#     cax.ax.tick_params(labelsize=10)
#     # plt.figtext(0.9125, 0.035, r'$\mathrm{lg}{n}$', fontsize=20)

#     f.subplots_adjust(wspace = 0.0, left=0.19, bottom=0.13, right=0.8, top=0.95)
#     # f.tight_layout()
#     #plt.savefig('./', format='png', dpi=300)
#     plt.show()

# def plot_jz(n, Vz, nframes, minvar, maxvar, figsizex = 18.0, figsizey = 8.0):
#     nt, nx, ny = n.shape
#     nf = len(nframes)

#     jz = n * Vz * 1.6e-23

#     x = np.linspace(0., 1., nx)
#     y = np.linspace(0., 1., ny)
#     Y, X  = np.meshgrid(y, x)

#     f = plt.figure(figsize=(figsizex/2.54, figsizey/2.54)) # plot size is defined in cms
#     wr = np.ones(nf)
#     gs = gridspec.GridSpec(1, nf, width_ratios=wr)

#     colorlev = np.linspace(minvar, maxvar, 100)
#     cmap = plt.cm.hot_r

#     for i in range(nf):
#         idt = nframes[i]
#         ax = f.add_subplot(gs[0, i])
#         if i == 0:
#             ax.set_ylabel(r'$z/L$', fontsize=12)
#         else:
#             plt.setp(ax.get_yticklabels(), visible = False)
#             ax.tick_params(axis = 'y', which = 'both', bottom = False, left = False)

#         ax.xaxis.set_major_locator(ticker.FixedLocator([0., 0.25, 0.5, 0.75]))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

#         cs = ax.contourf(X, Y, jz[idt, :, :],
#                          levels = colorlev,
#                          cmap = cmap,
#                          extend='both')

#         # ax.set_xlim([0., 0.5])
#         ax.set_xlabel(r'$r/R$', fontsize=12)
#         ax.set_title('$t\ =\ {%s}\ \mathrm{\mu{s}}$' % str(round(time[nframes[i]], 2)), fontsize = 12)

#     ax.xaxis.set_major_locator(ticker.FixedLocator([0., 0.25, 0.5, 0.75, 1.0]))

#     cbar_ax = f.add_axes([0.915, 0.1, 0.015, 0.825])
#     cax = f.colorbar(cs, cax=cbar_ax, ticks=np.arange(0.0, maxvar, 0.2), orientation = 'vertical', extend = 'both')
#     cax.ax.tick_params(labelsize=10)
#     plt.figtext(0.875, 0.05, r'$j_z,\ \mathrm{A/cm}^2$', fontsize=12)

#     f.subplots_adjust(wspace = 0.0, left=0.1, bottom=0.2, right=0.875)
#     # f.tight_layout()
#     #plt.savefig('./', format='png', dpi=300)
#     plt.show()

# def plot_jz_extractor(n, Vz, nframes, figsizex = 18.0, figsizey = 8.0):
#     nt, nx, ny = n.shape

#     jz = n * Vz * 1.6e-23

#     x = np.linspace(0., 1., nx)

#     plt.figure(figsize=(figsizex/2.54, figsizey/2.54))

#     for i in nframes:
#         plt.plot(x, jz[i, :, -1],
#                  label=r'$t\ =\ {%s}\ \mathrm{\mu{s}}$' % str(round(time[i], 2))
#                  )
#     plt.plot([extractor_r/Rcell, extractor_r/Rcell], [-105, 105], 'k--', label=r'Extr. boundary')
#     plt.xlim([0.,1.0])
#     plt.ylim([-2.0,12.0])
#     plt.xlabel(r'$r/R$', fontsize=12)
#     plt.ylabel(r'$j_z,\ \mathrm{A/cm^2}$', fontsize=12)
#     plt.legend(loc='best')

#     plt.show()

# def plot_avg_jz_at_yloc(n, Vz, nloc, tstart, tstop, figsizex = 18.0, figsizey = 8.0):
#     nt, nx, ny = n.shape

#     jz = n * Vz * 1.6e-23

#     x = np.linspace(0., Rcell, nx)
#     avg_jz = np.mean(jz[tstart:tstop,:,:], axis=0)

#     plt.figure(figsize=(figsizex/2.54, figsizey/2.54))

#     from scipy.integrate import trapezoid
#     current = trapezoid(avg_jz[:,-1]*x, x)*2.0*np.pi*1e3
#     print(current) # mA
#     # plt.figtext(0.7, 0.7, r'$I = %3.0f\ \mathrm{mA}$' % current)

#     for i in nloc:
#         plt.plot(x, avg_jz[:, i],
#                  # label=r'$z\ =\ {%s}\ \mathrm{m}$' % str(round(i/ny*Lcell, 2))
#                  label = r'$I = %3.0f\ \mathrm{mA}$' % current
#                  )
#     plt.plot([extractor_r, extractor_r], [-2.5, 5.5], 'k--', label=r'Extr. boundary')
#     plt.xlim([0.,Rcell])
#     plt.ylim([0.0,400e-3])
#     plt.xlabel(r'$r,\ \mathrm{cm}$', fontsize=12)
#     plt.ylabel(r'$\overline{j}_z,\ \mathrm{A/cm^2}$', fontsize=12)
#     plt.legend(loc='best')
#     plt.tight_layout()

#     plt.show()

# def plot_avg_phi_axis(n, Te, tstart, tstop, figsizex = 10.0, figsizey = 8.0):
#     """

#     Parameters
#     ----------
#     n : Ndarray
#         Plasma density.
#     Te : Ndarray
#         Plasma electron temperature.
#     tstart : int
#         Starting time index for averaging the potential.
#     tstop : int
#         Stopping time index for averaging the potential.
#     figsizex : TYPE, optional
#         DESCRIPTION. The default is 10.0.
#     figsizey : TYPE, optional
#         DESCRIPTION. The default is 8.0.

#     Returns
#     -------
#     None.

#     """
#     nt, nx, ny = n.shape

#     y = np.linspace(0., 1., ny)
#     phi_axis = np.ndarray((ny))

#     phi_axis = np.mean( cumulative_trapezoid( (np.gradient((n*Te),
#                                                axis=1,
#                                                edge_order=2) / n)[:,0,:] +
#                                                0.71 * np.gradient(Te[:,0,:],
#                                                                   axis=1,
#                                                                   edge_order=2),
#                                                axis=1, initial=0.0)[tstart:tstop,:],
#                        axis=0)
#     # print(phi_axis.shape)
#     # print(y.shape)
#     plt.figure(figsize=(figsizex/2.54,figsizey/2.54))
#     plt.plot(y, phi_axis, 'k', linewidth=1.5)
#     plt.plot(y, UnivariateSpline(y, phi_axis, s=1000)(y),
#              '--',
#              linewidth=1.5)
#     plt.xlim([0., 1.0])
#     plt.xlabel(r'$z/L$', fontsize=12)
#     plt.ylabel(r'$\overline{\varphi},\ \mathrm{V}$', fontsize=12)
#     plt.tight_layout()
#     plt.show()

# def plot_avg_prof_axis(f, varname, dimname, tstart, tstop, figsizex = 10.0, figsizey = 8.0):
#     """

#     Parameters
#     ----------
#     n : Ndarray
#         Plasma density.
#     Te : Ndarray
#         Plasma electron temperature.
#     tstart : int
#         Starting time index for averaging the potential.
#     tstop : int
#         Stopping time index for averaging the potential.
#     figsizex : TYPE, optional
#         DESCRIPTION. The default is 10.0.
#     figsizey : TYPE, optional
#         DESCRIPTION. The default is 8.0.

#     Returns
#     -------
#     None.

#     """
#     nt, nx, ny = f.shape

#     y = np.linspace(0., 1., ny)
#     f_axis = np.ndarray((ny))

#     f_axis = np.mean( f[tstart:tstop,0,:], axis=0)
#     # print(phi_axis.shape)
#     # print(y.shape)
#     plt.figure(figsize=(figsizex/2.54,figsizey/2.54))
#     plt.plot(y, f_axis, 'k', linewidth=1.5)
#     plt.xlim([0., 1.0])
#     plt.xlabel(r'$z/L$', fontsize=28)
#     plt.ylabel(r'$%s,\ \mathrm{' % varname + '%s}$' % dimname, fontsize=28)
#     plt.tight_layout()
#     plt.show()

# nframes = (13,25,50,200)
# nframes = (5, 10, 50, 90)
# plot_density(Ti, nframes, 0, 1.25, 16., 10.)
# plot_density(np.log10(n), nframes, 15, 19.5, 16., 16.)
# plot_density(Te, nframes, 0.5, 5.0, 16., 10.)
# plot_density(np.log10(n), nframes, 16.5, 19.5, 16., 16.)
# plot_density(n/1e19, nframes, 0., 1.75, 16., 16.)
# plot_density(np.log10(n**2.0*Te), nframes, 25., 40., 16., 16.)
# plot_density(np.log10(n), nframes, 18.0, 20., 16., 10.)
# plot_density(n, nframes, 1e19, 10e19)
# plot_avg_density(np.log10(n), 16.5, 18.5, 8., 10.)
# plot_jz(n, Vz, nframes, 0.0, 1.5, 16., 10.)
# nframes = np.arange(10, 40, 10)
# plot_jz_extractor(n, Vz, nframes, 10., 8.)
# plot_avg_jz_at_yloc(n[:51], Vz[:51], [99], 30, -1, 10., 8.)
# plot_avg_phi_axis(n, Te, 10, 80, figsizex = 10.0, figsizey = 8.0)
# plot_avg_prof_axis(n*1e-19, '\overline{n}', '10^{19}\ \mathrm{m}^3', 10, 80, figsizex = 10.0, figsizey = 8.0)
# plot_avg_prof_axis(Te, '\overline{T}_e', '\mathrm{eV}', 10, 80, figsizex = 10.0, figsizey = 8.0)


# плотность тока
# nt, nx, ny = n.shape
# jz = n * Vz * 1.6e-23
# nt, nx, ny = n.shape
# x = np.linspace(0., 1., nx)
# tmin = 10
# tmax = 50
# ymin = 0.0
# ymax = 10.0
# figsizex = 12.0
# figsizey = 9.0
# plt.figure(figsize=(figsizex/2.54, figsizey/2.54))
# plt.plot(x, np.mean(jz[tmin:tmax, :, -1], axis=0), label='Case 6')
# # plt.plot([extractor_r/Rcell, extractor_r/Rcell], [-105, 105], 'k--', label=r'Extr. boundary')
# plt.xlim([0.,0.75])
# # plt.ylim([ymin,ymax])
# plt.ylim([0.0,0.3])
# plt.xlabel(r'$r/R$', fontsize=12)
# plt.ylabel(r'$\overline{j}_z,\ \mathrm{A/cm^2}$', fontsize=12)
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()
