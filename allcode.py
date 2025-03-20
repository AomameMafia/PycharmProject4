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


# # для плотности плазмы, то что как раз нужно было
# nframes = 50  # Кадр, соответствующий 140 мкс

# # Загружаем данные по плотности
# n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Плотность в ячейке

# # **1. График плотности по оси z для 140 мкс**
# n_140us = n[nframes]  # Плотность для кадра, соответствующего 140 мкс
# n_z_profile = np.mean(n_140us, axis=0)  # Усредняем по радиальной координате

# # Переводим плотность в нормированные единицы 10^19 част/м^3
# n_z_profile = n_z_profile * 4.2e16  # Преобразуем, если требуется нормировка

# # Координаты по оси z
# z = np.linspace(0, Lcell, n_z_profile.shape[0])

# # **1. График плотности по оси z для 140 мкс**
# plt.figure(figsize=(16, 12))
# plt.plot(z, n_z_profile, label='Плотность плазмы n(z) при 140 мкс', linewidth=2)
# plt.axvline(x=cell_L, color='blue', linestyle='--', label='Начало накопительной ячейки 7.1 см')
# # plt.axvline(x=1.1, color='red', linestyle='--', label='Пик по плотности плазмы 1.1 см')

# # Подписи для оси z
# plt.xlabel('z, см', fontsize=28, fontweight='bold')
# plt.ylabel(r'Плотность плазмы × $10^{27}$ част/м$^3$', fontsize=28, fontweight='bold')

# Добавляем легенду с жирными подписями
# plt.legend(fontsize=28)

# # Ось z начинается с нуля, добавляем ограничение
# plt.xlim(0, Lcell)  # Ось z начинается с нуля
# plt.ylim(0)  # Ось плотности начинается с нуля

# # Подпишем точки на оси z
# plt.xticks(np.arange(0, Lcell+1, 2))  # Поставим шаг 1 см на оси z и наклоним метки
# plt.grid(False)
# plt.tick_params(axis='both', which='major', labelsize=24)

# # Отображаем график
# plt.show()


# def plot_density_by_radius(nframes, z_positions, Rcell, Lcell, cell_R, n_factor=1.0):
#     """
#     Построение графиков плотности плазмы по радиусу для заданных координат z.
#     """
#     # Загрузка плотности плазмы (например, из данных моделирования)
#     n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Файл с данными
#     n_frame = n[nframes] * n_factor  # Выбираем кадр и масштабируем

#     # Радиальные и продольные координаты
#     nx, ny = n_frame.shape
#     r = np.linspace(0, Rcell, nx)  # Радиальная координата
#     z = np.linspace(0, Lcell, ny)  # Продольная координата

#     # Построение графиков
#     plt.figure(figsize=(10, 6))
#     for z_pos in z_positions:
#         z_index = np.argmin(np.abs(z - z_pos))  # Находим индекс для текущего z
#         plt.plot(r, n_frame[:, z_index], label=f'z = {z_pos:.1f} см')

#     # Граница накопительной ячейки
#     plt.axvline(x=cell_R, color='red', linestyle='--', label='Граница накопительной ячейки')

#     # Настройка осей и легенды
#     plt.xlabel('Радиус r, мм', fontsize=14)
#     plt.ylabel('Плотность плазмы, × $10^{13}$ част/м³', fontsize=14)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.show()

# # Пример вызова
# plot_density_by_radius(nframes=200, z_positions=[3.5, 7.0, 14.0], Rcell=36, Lcell=286, cell_R=9, n_factor=4.2e-16)


# сразу отработка куча папок дял плотности плазмы
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# # from your_module import collect  # Импорт функции для загрузки данных

# # Генерация списка папок
# base_path = r'C:\Users\auezo\Desktop\fromcluster'
# data_dirs = [os.path.join(base_path, f'Test{i}') for i in range(1, 63)]  # Список от Test1 до Test62

# # Список для проблемных папок
# failed_dirs = []

# def plot_density_z_for_directory(data_dir, nframes, Lcell, cell_L):
#     """
#     Построить график плотности по оси z для указанной папки.

#     Parameters:
#     - data_dir: str, путь к папке с данными
#     - nframes: int, кадр для построения графика
#     - Lcell: float, длина ячейки в см
#     - cell_L: float, начало накопительной ячейки
#     """
#     os.chdir(data_dir)  # Переход в папку с данными

#     # Загружаем плотность
#     n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Плотность в ячейке

#     # Проверка максимального доступного индекса
#     max_index = n.shape[0] - 1  # Максимальный индекс по оси времени
#     if nframes > max_index:
#         print(f"Предупреждение: nframes={nframes} превышает доступный размер {max_index + 1}. Используется последний кадр.")
#         nframes = max_index  # Выбираем последний доступный кадр

#     # Плотность для выбранного кадра
#     n_140us = n[nframes]
#     n_z_profile = np.mean(n_140us, axis=0)  # Усредняем по радиальной координате

#     # Переводим плотность в нормированные единицы 10^19 част/м^3
#     n_z_profile = n_z_profile * 4.2e16

#     # Координаты по оси z
#     z = np.linspace(0, Lcell, n_z_profile.shape[0])

#     # Построение графика
#     plt.figure(figsize=(16, 12))
#     plt.plot(z, n_z_profile, label=f'Плотность плазмы n(z), папка: {os.path.basename(data_dir)}', linewidth=2)
#     plt.axvline(x=cell_L, color='blue', linestyle='--', label='Начало накопительной ячейки 7.1 см')
#     plt.axvline(x=1.1, color='red', linestyle='--', label='Пик по плотности плазмы 1.1 см')

#     # Настройка графика
#     plt.title(f'График плотности для {os.path.basename(data_dir)}', fontsize=28, fontweight='bold')
#     plt.xlabel('z, см', fontsize=28, fontweight='bold')
#     plt.ylabel(r'Плотность плазмы × $10^{13}$ част/м$^3$', fontsize=28, fontweight='bold')
#     plt.xlim(0, Lcell)
#     plt.ylim(0)
#     plt.xticks(np.arange(0, Lcell + 1, 2))
#     plt.grid(False)
#     plt.tick_params(axis='both', which='major', labelsize=24)
#     plt.legend(fontsize=20)
#     plt.tight_layout()
#     plt.show()

# # Цикл по всем папкам
# for data_dir in data_dirs:
#     folder_name = os.path.basename(data_dir)  # Имя папки, например Test22
#     try:
#         plot_density_z_for_directory(
#             data_dir=data_dir,
#             nframes=50,  # Пример: кадр, соответствующий 140 мкс
#             Lcell=20.0,  # Пример: длина ячейки
#             cell_L=7.1  # Пример: начало накопительной ячейки
#         )
#     except (OSError, FileNotFoundError, IndexError) as e:
#         print(f"Ошибка с папкой {folder_name}: {e}")
#         failed_dirs.append(folder_name)

# # Вывод всех проблемных папок
# if failed_dirs:
#     print("\nПроблемы возникли с папками:")
#     for folder in failed_dirs:
#         print(f"- {folder}")
# else:
#     print("\nВсе папки обработаны успешно!")

import os
import numpy as np
import matplotlib.pyplot as plt

# from your_module import collect  # Импорт функции для загрузки данных

# Список папок для обработки
specific_dirs = [
    r'C:\Users\Alisher\Desktop\fromcluster\Test55',
    # r'C:\Users\auezo\Desktop\fromcluster\Test16',
    # r'C:\Users\auezo\Desktop\fromcluster\Test17'
]

# Пользовательская легенда для графиков
user_legends = {
    "Test55": "N(4) = 200, I(4) = 700 A",
    # "Test16": "1E+20 част/м^3",
    # "Test17": "1E+21 част/м^3"
}

# Список для проблемных папок
failed_dirs = []

# Стили маркеров для графиков (разные маркеры для ч/б режима)
markers = ['o', '^', 's', 'D']  # Кружок, треугольник, квадрат, ромб


def plot_density_z_combined(specific_dirs, user_legends, nframes, Lcell, cell_L):
    """
    Построить комбинированный график плотности по оси z для указанных папок.

    Parameters:
    - specific_dirs: list, список путей к папкам с данными
    - user_legends: dict, словарь с пользовательскими подписями для легенд
    - nframes: int, кадр для построения графика
    - Lcell: float, длина ячейки в см
    - cell_L: float, начало накопительной ячейки
    """
    plt.figure(figsize=(16, 12))

    for idx, data_dir in enumerate(specific_dirs):
        folder_name = os.path.basename(data_dir)  # Имя папки
        try:
            os.chdir(data_dir)  # Переход в папку с данными

            # Загружаем плотность
            n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]

            # Проверка максимального доступного индекса
            max_index = n.shape[0] - 1
            if nframes > max_index:
                print(
                    f"Предупреждение: nframes={nframes} превышает доступный размер {max_index + 1}. Используется последний кадр.")
                nframes = max_index

            # Плотность для выбранного кадра
            n_140us = n[nframes]
            n_z_profile = np.mean(n_140us, axis=0)

            # Переводим плотность в нормированные единицы 10^18 част/м^3
            n_z_profile = n_z_profile * 4.2e16  # Изменён масштаб

            # Координаты по оси z
            z = np.linspace(0, Lcell, n_z_profile.shape[0])

            # Добавляем график с маркерами
            plt.plot(
                z, n_z_profile,
                label=user_legends.get(folder_name, f"График для {folder_name}"),
                marker=markers[idx % len(markers)],  # Выбор маркера
                markersize=8,  # Размер маркеров
                linewidth=2
            )

        except (OSError, FileNotFoundError, IndexError) as e:
            print(f"Ошибка с папкой {folder_name}: {e}")
            failed_dirs.append(folder_name)

    # Настройка графика
    plt.axvline(x=cell_L, color='blue', linestyle='--', label='Начало накопительной ячейки 7.1 см')
    # plt.axvline(x=1.1, color='red', linestyle='--', label='Пик по плотности плазмы 1.1 см')
    # plt.title('Объединённый график плотности плазмы', fontsize=28, fontweight='bold')
    plt.xlabel('z, см', fontsize=28, fontweight='bold')
    plt.ylabel(r'Плотность плазмы × $10^{18}$ част/м$^3$', fontsize=28, fontweight='bold')
    plt.xlim(0, Lcell)
    plt.ylim(0)
    plt.xticks(np.arange(0, Lcell + 1, 2))
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()


# Вызов функции для построения объединённого графика
plot_density_z_combined(
    specific_dirs=specific_dirs,
    user_legends=user_legends,
    nframes=50,  # Пример: кадр, соответствующий 140 мкс
    Lcell=28.6,  # Пример: длина ячейки
    cell_L=7.1  # Пример: начало накопительной ячейки
)

# Вывод всех проблемных папок
if failed_dirs:
    print("\nПроблемы возникли с папками:")
    for folder in failed_dirs:
        print(f"- {folder}")
else:
    print("\nВсе папки обработаны успешно!")

# Аналогично предыдущему коду, но для температуры
nframes = 62  # Кадр, соответствующий 140 мкс

# Загружаем данные по температуре
Ti = collect('Ti', yguards=True)[:, 2:-2, 2:-2, 0]  # Температура ионов
Te = collect('Te', yguards=True)[:, 2:-2, 2:-2, 0]  # Температура электронов

# Температуры для кадра, соответствующего 140 мкс
Ti_140us = Ti[nframes]
Te_140us = Te[nframes]

# Усредняем по радиальной координате
Ti_z_profile = 0.03 * np.mean(Ti_140us, axis=0)
Te_z_profile = 0.03 * np.mean(Te_140us, axis=0)
# Координаты по оси z
z = np.linspace(0, Lcell, Ti_z_profile.shape[0])

# Построение графика
plt.figure(figsize=(10, 6))

# График температуры ионов
plt.plot(z, Ti_z_profile, label='Температура ионов', color='red', linewidth=2)

# График температуры электронов
plt.plot(z, Te_z_profile, label='Температура электронов', color='blue', linewidth=2)

# Вспомогательные линии
plt.axvline(x=cell_L, color='green', linestyle='--', label='Начало накопительной ячейки 7.1 см')

# Настройка осей
plt.xlabel('z, см', fontsize=16, fontweight='bold')
plt.ylabel('Температура, эВ', fontsize=16, fontweight='bold')
plt.title('Распределение температуры ионов и электронов по оси z в момент 140 мкс', fontsize=14, fontweight='bold')

# Настройка осей и легенды
plt.xlim(0, Lcell)
plt.ylim(0)
plt.xticks(np.arange(0, Lcell + 1, 2))
plt.legend(fontsize=12)

plt.show()

plt.figure(figsize=(10, 6))
# Ti_mean = np.mean(Ti, axis=(1,2))
# Te_mean = np.mean(Te, axis=(1,2))

# plt.plot(time, Ti_mean, label='Средняя температура ионов')
# plt.plot(time, Te_mean, label='Средняя температура электронов')
# plt.title('Эволюция средних температур')
# plt.xlabel('Время, мкс')
# plt.ylabel('Температура, эВ')
# plt.legend()
# plt.show()


# Корреляция плотности и температуры IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed

# <Figure size 720x432 with 0 Axes>

# plt.figure(figsize=(10, 6))
# plt.scatter(n[nframes].flatten(), Ti[nframes].flatten(),
#             alpha=0.5, label='Ионы')
# plt.scatter(n[nframes].flatten(), Te[nframes].flatten(),
#             alpha=0.5, label='Электроны')
# plt.xlabel('Плотность')
# plt.ylabel('Температура')
# plt.title('Корреляция плотности и температуры')
# plt.legend()
# plt.show()


# plt.figure(figsize=(12, 8))
# plt.imshow(n[nframes], aspect='auto', cmap='viridis')
# plt.colorbar(label='Плотность')
# plt.title('2D распределение плотности в момент 140 мкс')
# plt.show()

# Lcell = 2.86e-1  # plasma cell length [m]
# Rcell = 3.6e-2  # plasma cell radius [m]

# cell_L = Lcell/4.0

# nframes = 200  # Кадр, соответствующий 140 мкс

# # Данные
# data_2d = n[nframes]

# # Расчетная область
# z = np.linspace(0, Lcell, data_2d.shape[0])
# r = np.linspace(0, Rcell, data_2d.shape[1])

# # Построение графика
# plt.figure(figsize=(10, 8))
# plt.contourf(z*100, r*100/2, data_2d.T, levels=20, cmap='inferno')
# plt.colorbar(label='Плотность, отн. ед.')

# # Вертикальная линия начала накопительной ячейки
# plt.axvline(x=cell_L*100, color='white', linestyle='--', label='Начало накопительной ячейки')

# plt.title('2D профиль плотности плазмы в момент 140 мкс')
# plt.xlabel('Продольная координата z, см')
# plt.ylabel('Радиальная координата r, см')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # плотность плазмы
# nframes = 50

# # Данные
# data_2d = n[nframes] * 4.2e16  # Перевод в част/м^3

# # Расчетная область
# z = np.linspace(0, Lcell, data_2d.shape[0])
# r = np.linspace(0, Rcell, data_2d.shape[1])
# nframes = 50

# # Данные
# data_2d = n[nframes]

# # Расчетная область
# z = np.linspace(0, Lcell, data_2d.shape[0])
# r = np.linspace(0, Rcell, data_2d.shape[1])

# # Построение графика
# plt.figure(figsize=(10, 8))
# plt.contourf(z*100, r*100/2, data_2d.T, levels=20, cmap='inferno')
# plt.colorbar(label='Плотность, отн. ед.')

# # Вертикальная линия начала накопительной ячейки
# plt.axvline(x=cell_L*100, color='white', linestyle='--', label='Начало накопительной ячейки')

# plt.title('2D профиль плотности плазмы в момент 140 мкс')
# plt.xlabel('Продольная координата z, см')
# plt.ylabel('Радиальная координата r, см')
# plt.legend()

# plt.tight_layout()
# plt.show()


# рапсределение МП интегрирование по току только 286

# from scipy.integrate import cumulative_trapezoid


# # Данные плотности тока (примерные)
# z = np.linspace(0, Lcell, 1000)  # координаты вдоль оси z
# jz = np.exp(-((z - Lcell/2) / (Lcell/10))**2) * 10  # модельное распределение плотности тока jz (A/см^2)

# # Вычисление магнитного поля Bz вдоль оси z по закону Био-Савара
# mu_0 = 4 * np.pi * 1e-7 * 1e4 * 10 # магнитная постоянная (Гс·см/А)
# Bz = mu_0 * cumulative_trapezoid(jz, z, initial=0)  # поле Bz вдоль оси z

# # Построение графика
# plt.figure(figsize=(10, 6))
# plt.plot(z, Bz, label=r'$B_z(z)$', color='blue', linewidth=2)
# plt.axvline(x=7.1, color='green', linestyle='--', label='Начало накопительной ячейки (7.1 см)')
# plt.axvline(x=Lcell, color='red', linestyle='--', label='Конец накопительной ячейки (286 см)')

# # Настройка графика
# plt.title('Распределение магнитного поля $B_z$ вдоль оси z', fontsize=14, fontweight='bold')
# plt.xlabel('Координата z, см', fontsize=12)
# plt.ylabel('Магнитное поле $B_z$, кГс', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# Магнинте поле попытка построить как надо
# import matplotlib.pyplot as plt
# from scipy.integrate import cumulative_trapezoid
# import numpy as np

# # Параметры
# Lcell = 28.6 # конец накопительной ячейки
# z1 = np.linspace(0, Lcell, 1000) # координаты до конца ячейки
# z2 = np.linspace(Lcell, 35, 500) # координаты области спада
# z3 = np.linspace(35, 40, 200) # координаты области перехода

# # Данные плотности тока для накопительной ячейки
# jz1 = np.exp(-((z1 - Lcell/2) / (Lcell/10))**2) * 10

# # Вычисление магнитного поля в накопительной ячейке
# mu_0 = 4 * np.pi * 1e-7 * 1e4 * 10
# Bz1 = cumulative_trapezoid(jz1, z1, initial=0)

# # Нормировка, чтобы максимальное поле было 6 кГс
# Bz1 = Bz1 / np.max(Bz1) * 3

# # Значение поля в конце накопительной ячейки
# B_end = Bz1[-1]

# # Экспоненциальный спад от конца накопительной ячейки до 1.3 кГс
# Bz2 = (B_end - 1.3) * np.exp(-((z2 - Lcell) / ((35 - Lcell)/3))**2) + 1.3

# # Создание ровного участка постоянного поля в районе 35 см
# Bz3 = np.full_like(z3, 1.3)

# # Объединение массивов для построения полного графика
# z = np.concatenate([z1, z2, z3])
# Bz = np.concatenate([Bz1, Bz2, Bz3])

# # Настройка шрифтов и размеров
# plt.rcParams.update({'font.size': 14}) # Базовый размер шрифта
# plt.figure(figsize=(14, 10)) # Увеличенный размер графика с местом для легенды

# plt.plot(z, Bz, label=r'$B_z(z)$', color='blue', linewidth=2)
# plt.axvline(x=7.1, color='green', linestyle='--', label='Начало накопительной ячейки, 7.1 см')
# plt.axvline(x=Lcell, color='red', linestyle='--', label='Конец накопительной ячейки, 28.6 см ')
# plt.axvline(x=35, color='orange', linestyle='--', label='Начало постоянного поля, 35 см')
# plt.axhline(y=1.3, color='purple', linestyle=':', label='B(z) = 1.3 кГс')

# plt.xlabel('z, см', fontsize=28)
# plt.ylabel('Магнитное поле $B_z$, кГс', fontsize=28)

# # Увеличение размера делений осей
# plt.tick_params(axis='both', which='major', labelsize=20)

# plt.ylim(0) # Начало оси Y с нуля
# plt.xlim(0) # Начало оси X с нуля

# # Размещение легенды под графиком
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#            ncol=2, fontsize=20, frameon=True)

# plt.tight_layout(rect=[0, 0.1, 1, 1])  # Оставляем место для легенды
# plt.show()


nframes = 5
# def plot_temperature_by_radius(nframes, z_positions, Rcell, Lcell, cell_R, temp_type='Te', temp_factor=1.0):
#     """
#     Построить графики распределения температуры по радиусу на заданных расстояниях z.

#     Parameters:
#     - nframes: int, индекс временного кадра
#     - z_positions: list, координаты z (в см), на которых строится распределение
#     - Rcell: float, радиус ячейки в см
#     - Lcell: float, длина ячейки в см
#     - cell_R: float, граница накопительной ячейки в см
#     - temp_type: str, тип температуры ('Te' - электронная, 'Ti' - ионная)
#     - temp_factor: float, множитель для масштабирования температуры

#     Returns:
#     - None
#     """
#     # Загружаем температуру
#     temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]  # 'Te' или 'Ti'
#     temp_140us = temp_data[nframes] * temp_factor  # Берем кадр и масштабируем

#     # Координаты
#     nx, ny = temp_140us.shape
#     r = np.linspace(0, Rcell, nx)  # Радиальная координата
#     z = np.linspace(0, Lcell, ny)  # Продольная координата

#     # Построение графиков
#     plt.figure(figsize=(10, 6))
#     for z_pos in z_positions:
#         z_index = np.argmin(np.abs(z - z_pos))  # Ищем ближайший индекс
#         plt.plot(r, temp_140us[:, z_index], label=f'z = {z_pos:.1f} см')

#     # Добавляем границу накопительной ячейки
#     plt.axvline(x=cell_R, color='red', linestyle='--', label='Граница накопительной ячейки')

#     # Настройка графика
#     plt.xlabel('Радиус r, см', fontsize=28, fontweight='bold')
#     plt.ylabel('Температура, эВ', fontsize=28, fontweight='bold')
#     # plt.legend(fontsize=20)


#     plt.xlim(0, Rcell)  # Устанавливаем начало оси X с нуля
#     plt.ylim(0)  # Устанавливаем начало оси X с нуля
#     plt.grid(False)
#     plt.tick_params(axis='both', which='major', labelsize=20)

#     plt.tight_layout()
#     plt.show()


# # Пример вызова функции:
# plot_temperature_by_radius(nframes=5, z_positions=[30,40], Rcell=3.6, Lcell=28.6, cell_R=0.9, temp_type='Te', temp_factor=3.0e-2)


# nframes=5


# def plot_density_by_diameter(nframes, z_positions, Rcell, Lcell, cell_R, n_factor=1.0):
#     """
#     Построить графики распределения плотности плазмы по диаметру на заданных расстояниях z.

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
#     n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Из файла, уже настроено
#     n_140us = n[nframes] * n_factor  # Берем кадр и масштабируем

#     # Координаты
#     nx, ny = n_140us.shape
#     d = np.linspace(-Rcell, Rcell, 2 * nx)  # Диаметр: от -Rcell до Rcell
#     z = np.linspace(0, Lcell, ny)          # Продольная координата

#     # Построение графиков
#     plt.figure(figsize=(12, 8))
#     for z_pos in z_positions:
#         z_index = np.argmin(np.abs(z - z_pos))  # Ищем ближайший индекс
#         profile = np.concatenate([n_140us[::-1, z_index], n_140us[:, z_index]])
#         plt.plot(d, profile, label=f'z = {z_pos:.1f} см')

#     # Добавляем границы накопительной ячейки
#     plt.axvline(x=-cell_R, color='red', linestyle='--', label='Левая граница накопительной ячейки')
#     plt.axvline(x=cell_R, color='red', linestyle='--', label='Правая граница накопительной ячейки')

#     # Настройка графика
#     plt.title('Распределение плотности плазмы по диаметру', fontsize=18, fontweight='bold')
#     plt.xlabel('Диаметр d, см', fontsize=16, fontweight='bold')
#     plt.ylabel('Плотность плазмы, × $10^{13}$ част/м³', fontsize=16, fontweight='bold')
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.legend(fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()


# # Пример вызова функции (адаптируйте параметры для анализа):
# plot_density_by_diameter(nframes=50, z_positions=[7.0, 14.0], Rcell=3.6, Lcell=28.6, cell_R=0.9, n_factor=4.2e16)
# # nframes=50
# def plot_temperature_section(nframes, Rcell, Lcell, temp_type='Ti', temp_factor=1.0,  figsizex=18.0, figsizey=8.0):
#     """
#     Построить 2D-сечение распределения температуры в ячейке.

#     Parameters:
#     - nframes: int, индекс временного кадра
#     - Rcell: float, радиус ячейки в см
#     - Lcell: float, длина ячейки в см
#     - temp_type: str, тип температуры ('Te' - электронная, 'Ti' - ионная)
#     - temp_factor: float, множитель для масштабирования температуры

#     Returns:
#     - None
#     """
#     # Загружаем данные температуры
#     temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]  # 'Te' или 'Ti'
#     temp_140us = temp_data[nframes] * temp_factor  # Берем кадр и масштабируем

#     # Координаты
#     nx, ny = temp_140us.shape
#     r = np.linspace(0, Rcell, nx)  # Радиальная координата
#     z = np.linspace(0, Lcell, ny)  # Продольная координата

#     # Построение 2D-графика

#     plt.figure(figsize=(figsizex / 2.54, figsizey / 2.54))  # plot size is defined in cms
#     plt.contourf(r, z, temp_140us.T, levels=50, cmap='inferno')  # Транспонируем массив для правильной ориентации
#     plt.colorbar(label=f'{temp_type}, эВ')

#     # Граница накопительной ячейки
#     plt.axvline(x=Rcell, color='blue', linestyle='--', label='Граница накопительной ячейки')
#     plt.axhline(y=Lcell / 4, color='green', linestyle='--', label='Сечение накопительной ячейки')

#     # Настройка графика
#     plt.title(f'Распределение {temp_type} в пространстве', fontsize=14, fontweight='bold')
#     plt.xlabel('Радиус r, см', fontsize=12, fontweight='bold')
#     plt.ylabel('Координата z, см', fontsize=12, fontweight='bold')
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.show()


# малоинформативный график температуры
# def plot_temperature_cross_section_fixed_z(nframes, z_pos, Rcell, num_theta=100, temp_type='Te', temp_factor=1.0):
#     """
#     Построить 2D-сечение температуры в полярной системе координат (r, θ) на фиксированном z.

#     Parameters:
#     - nframes: int, индекс временного кадра
#     - z_pos: float, расстояние z (в см), на котором строится поперечное сечение
#     - Rcell: float, радиус ячейки в см
#     - num_theta: int, количество угловых разбиений для θ
#     - temp_type: str, тип температуры ('Te' - электронная, 'Ti' - ионная)
#     - temp_factor: float, множитель для масштабирования температуры

#     Returns:
#     - None
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Загружаем данные температуры
#     temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]
#     temp_140us = temp_data[nframes] * temp_factor  # Берем кадр и масштабируем

#     # Координаты
#     nx, ny = temp_140us.shape
#     r = np.linspace(0, Rcell, nx)  # Радиальная координата
#     z = np.linspace(0, Rcell, ny)  # Продольная координата

#     # Находим индекс для z_pos
#     z_index = np.argmin(np.abs(z - z_pos))
#     temp_at_z = temp_140us[:, z_index]  # Срез по температуре на z_pos

#     # Генерация координат для 2D-сечения
#     theta = np.linspace(0, 2 * np.pi, num_theta)  # Угловая координата
#     r_grid, theta_grid = np.meshgrid(r, theta)
#     temp_2d = np.tile(temp_at_z, (num_theta, 1))  # Повторяем профиль для всех углов

#     # Преобразование в декартовы координаты для построения
#     x = r_grid * np.cos(theta_grid)
#     y = r_grid * np.sin(theta_grid)

#     # Построение графика
#     plt.figure(figsize=(8, 8))
#     plt.contourf(x, y, temp_2d, levels=50, cmap='inferno')
#     plt.colorbar(label=f'{temp_type}, эВ')

#     # Настройка графика
#     plt.title(f'Сечение {temp_type} на z = {z_pos:.1f} см', fontsize=14, fontweight='bold')
#     plt.xlabel('Координата x, см', fontsize=12, fontweight='bold')
#     plt.ylabel('Координата y, см', fontsize=12, fontweight='bold')
#     plt.axis('equal')  # Равные масштабы осей
#     plt.tight_layout()
#     plt.show()


# # Пример вызова:
# plot_temperature_cross_section_fixed_z(nframes=50, z_pos=7.0, Rcell=3.6, num_theta=100, temp_type='Te', temp_factor=3.0e-2)


# # ValueError: too many values to unpack (expected 2)
# def plot_temperature_section_from_file(nframes, z_pos, Rcell, Lcell, temp_type='Ti', temp_factor=1.0):
#     """
#     Построить 2D-сечение температуры в пространстве, адаптированное из исходного кода.

#     Parameters:
#     - nframes: int, индекс временного кадра
#     - z_pos: float, расстояние z (в см), на котором строится поперечное сечение
#     - Rcell: float, радиус ячейки в см
#     - Lcell: float, длина ячейки в см
#     - temp_type: str, тип температуры ('Te' - электронная, 'Ti' - ионная)
#     - temp_factor: float, множитель для масштабирования температуры

#     Returns:
#     - None
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib import ticker, gridspec

#     # Загружаем данные температуры
#     temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]
#     temp_140us = temp_data[nframes] * temp_factor  # Берем кадр и масштабируем

#     # Координаты
#     nx, ny = temp_140us.shape
#     r = np.linspace(0, Rcell, nx)  # Радиальная координата
#     z = np.linspace(0, Lcell, ny)  # Продольная координата

#     # Находим индекс для z_pos
#     z_index = np.argmin(np.abs(z - z_pos))
#     temp_at_z = temp_140us[:, z_index]  # Срез по температуре на z_pos

#     # Генерация 2D-сетки (радиус и угловая координата)
#     theta = np.linspace(0, 2 * np.pi, nx)  # Угловая координата
#     r_grid, theta_grid = np.meshgrid(r, theta)
#     temp_2d = np.tile(temp_at_z, (nx, 1))  # Повторяем профиль для всех углов

#     # Преобразование в декартовы координаты
#     x = r_grid * np.cos(theta_grid)
#     y = r_grid * np.sin(theta_grid)

#     # Построение графика
#     fig = plt.figure(figsize=(10, 8))
#     gs = gridspec.GridSpec(1, 1)
#     ax = fig.add_subplot(gs[0, 0])

#     cs = ax.contourf(x, y, temp_2d, levels=100, cmap='inferno', extend='both')
#     cbar = fig.colorbar(cs, ax=ax, orientation='vertical', label=f'{temp_type}, эВ')
#     cbar.ax.tick_params(labelsize=10)

#     # Настройка графика
#     ax.set_title(f'Поперечное сечение {temp_type} на z = {z_pos:.1f} см', fontsize=14, fontweight='bold')
#     ax.set_xlabel('Координата x, см', fontsize=12)
#     ax.set_ylabel('Координата y, см', fontsize=12)
#     ax.axis('equal')
#     plt.tight_layout()
#     plt.show()


# # Пример вызова:
# plot_temperature_section_from_file(nframes=[50,100,150,200], z_pos=28.0, Rcell=3.6, Lcell=28.6, temp_type='Ti', temp_factor=3.0e-2)

# Не нужный график с сечением температуры на радиусе
# def plot_temperature_sections_from_file(nframes, z_positions, Rcell, Lcell, temp_type='Ti', temp_factor=1.0):
#     """
#     Построить 2D-сечения температуры в пространстве для нескольких z-позиций.

#     Parameters:
#     - nframes: int, индекс временного кадра
#     - z_positions: list of float, расстояния z (в см), на которых строятся поперечные сечения
#     - Rcell: float, радиус ячейки в см
#     - Lcell: float, длина ячейки в см
#     - temp_type: str, тип температуры ('Te' - электронная, 'Ti' - ионная)
#     - temp_factor: float, множитель для масштабирования температуры

#     Returns:
#     - None
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib import ticker, gridspec

#     # Загружаем данные температуры
#     temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]
#     temp_140us = temp_data[nframes] * temp_factor  # Берем кадр и масштабируем

#     # Координаты
#     nx, ny = temp_140us.shape
#     r = np.linspace(0, Rcell, nx)  # Радиальная координата
#     z = np.linspace(0, Lcell, ny)  # Продольная координата

#     # Создаем фигуру с подграфиками
#     n_plots = len(z_positions)
#     fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6), subplot_kw={'aspect': 'equal'})

#     if n_plots == 1:
#         axes = [axes]  # Преобразуем в список для универсальной обработки

#     for ax, z_pos in zip(axes, z_positions):
#         # Находим индекс для z_pos
#         z_index = np.argmin(np.abs(z - z_pos))
#         temp_at_z = temp_140us[:, z_index]  # Срез по температуре на z_pos

#         # Генерация 2D-сетки (радиус и угловая координата)
#         theta = np.linspace(0, 2 * np.pi, nx)  # Угловая координата
#         r_grid, theta_grid = np.meshgrid(r, theta)
#         temp_2d = np.tile(temp_at_z, (nx, 1))  # Повторяем профиль для всех углов

#         # Преобразование в декартовы координаты
#         x = r_grid * np.cos(theta_grid)
#         y = r_grid * np.sin(theta_grid)

#         # Построение графика
#         cs = ax.contourf(x, y, temp_2d, levels=20, cmap='inferno', extend='both')
#         ax.set_title(f'z = {z_pos:.1f} см', fontsize=14, fontweight='bold')
#         ax.set_xlabel('x, см', fontsize=16)
#         ax.set_ylabel('y, см', fontsize=16)

#         # Добавляем цветовую шкалу
#         cbar = fig.colorbar(cs, ax=ax, orientation='vertical', label=f'{temp_type}, эВ')
#         cbar.ax.tick_params(labelsize=14)

#     # Общая настройка графика
#     plt.tight_layout()
#     plt.show()


# # Пример вызова:
# plot_temperature_sections_from_file(
#     nframes=50,
#     z_positions=[3.5,7.0, 14.0, 21, 28.0],
#     Rcell=3.6,
#     Lcell=28.6,
#     temp_type='Ti',
#     temp_factor=3.0e-2
# )


# ***************НУЖНЫЙ ГРАФИК
# def plot_density1(n, nframes, minvar, maxvar, figsizex=18.0, figsizey=8.0):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from matplotlib import gridspec
#     from matplotlib import ticker

#     nt, nx, ny = n.shape
#     nf = len(nframes)

#     x = np.linspace(0., Rcell, nx)
#     y = np.linspace(0., Lcell, ny)
#     Y, X = np.meshgrid(y, x)

#     f = plt.figure(figsize=(figsizex / 2.54, figsizey / 2.54))  # plot size is defined in cms
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
#             plt.setp(ax.get_yticklabels(), visible=False)
#             ax.tick_params(axis='y', which='both', bottom=False, left=False)

#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
#         ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
#         ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

#         cs = ax.contourf(X, Y, n[idt, :, :],
#                           levels=colorlev,
#                           cmap=cmap,
#                           extend='both')
#         ax.plot([cell_r, cell_R], [cell_L, cell_L], 'b', linewidth=2.0)
#         ax.plot([cell_R, cell_R], [cell_L, Lcell], 'b', linewidth=2.0)

#         # Display the integer number for each time point
#         ax.set_title('%d' % int(round(time[nframes[i]])), fontsize=20, pad=20)

#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
#     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

#     cbar_ax = f.add_axes([0.9, 0.2, 0.015, 0.675])
#     cax = f.colorbar(cs, cax=cbar_ax, ticks=np.linspace(minvar, maxvar, 5), orientation='vertical', extend='both')

#     # Format colorbar ticks as integers
#     cax.ax.set_yticklabels(['%d' % t for t in np.linspace(minvar, maxvar, 5)])

#     cax.ax.tick_params(labelsize=10)

#     # Set central xlabel across all plots
#     f.text(0.5, 0.04, r'$r,\ \mathrm{cm}$', ha='center', fontsize=28)

#     # Set the overall figure title slightly to the further left
#     f.suptitle(r'$t,\ \mathrm{\mu{s}}$:', fontsize=24, x=0.05, y=0.98)

#     # Adjust spacing to avoid overlap
#     f.subplots_adjust(wspace=0.3, left=0.1, bottom=0.15, right=0.875, top=0.9)

#     plt.show()


# nframes = (12,25,50,200)
# plot_density1(np.log10(n), nframes, 15, 19.5, 16., 16.)


#  4 картинки температуры ионов в разный момент времени
# def plot_temperature_section(nframes, Rcell, Lcell, temp_type='Ti', temp_factor=1.0, figsizex=18.0, figsizey=8.0):
#     """
#     Построить 2D-сечение распределения температуры в ячейке для нескольких моментов времени.

#     Parameters:
#     - nframes: iterable, индексы временных кадров
#     - Rcell: float, радиус ячейки в см
#     - Lcell: float, длина ячейки в см
#     - temp_type: str, тип температуры ('Te' - электронная, 'Ti' - ионная)
#     - temp_factor: float, множитель для масштабирования температуры

#     Returns:
#     - None
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from matplotlib import gridspec

#     # Загружаем данные температуры
#     temp_data = collect(temp_type, yguards=True)[:, 2:-2, 2:-2, 0]  # 'Te' или 'Ti'

#     # Создаем фигуру и сетку для подграфиков
#     num_frames = len(nframes)
#     f = plt.figure(figsize=(figsizex / 2.54 * num_frames, figsizey / 2.54))
#     gs = gridspec.GridSpec(1, num_frames, width_ratios=[1]*num_frames)

#     nx, ny = temp_data[0, :, :].shape
#     r = np.linspace(0, Rcell, nx)  # Радиальная координата
#     z = np.linspace(0, Lcell, ny)  # Продольная координата

#     # Определяем диапазон для цветовой шкалы
#     min_temp = (temp_data.min() * temp_factor)
#     max_temp = (temp_data.max() * temp_factor)

#     for idx, frame in enumerate(nframes):
#         temp_frame = temp_data[frame] * temp_factor  # Берем кадр и масштабируем

#         # Настройка подграфика
#         ax = f.add_subplot(gs[idx])
#         contour = ax.contourf(r, z, temp_frame.T, levels=50, cmap='inferno', vmin=min_temp, vmax=max_temp)

#         # Добавить синие линии, обозначающие границы ячейки
#         ax.plot([0, Rcell], [Lcell / 4, Lcell / 4], 'b', linewidth=2.0)
#         ax.plot([Rcell, Rcell], [0, Lcell / 4], 'b', linewidth=2.0)

#         # Настройка заголовка подграфика
#         ax.set_title(f'Время: {frame} мкс', fontsize=12, fontweight='bold')
#         ax.set_xlabel('Радиус r, см', fontsize=10, fontweight='bold')

#         if idx == 0:
#             ax.set_ylabel('Координата z, см', fontsize=10, fontweight='bold')
#         else:
#             plt.setp(ax.get_yticklabels(), visible=False)

#         ax.tick_params(axis='both', which='major', labelsize=8)

#     # Добавляем общую цветовую шкалу для всех подграфиков
#     cbar = f.colorbar(contour, ax=f.get_axes(), orientation='vertical', fraction=0.02, pad=0.04)
#     cbar.set_label(f'{temp_type}, эВ', fontsize=12, fontweight='bold')
#     cbar.ax.tick_params(labelsize=10)

#     # Форматирование меток цветовой шкалы без лишних десятичных знаков
#     cbar.set_ticks([min_temp, (min_temp + max_temp) / 2, max_temp])
#     cbar.set_ticklabels(['%d' % t for t in [min_temp, (min_temp + max_temp) / 2, max_temp]])

#     plt.tight_layout()
#     plt.show()

# # Пример вызова функции
# nframes = (12, 25, 50, 200)
# plot_temperature_section(nframes, Rcell=3.6, Lcell=28.6, temp_type='Ti', temp_factor=3.0e-2)


# тоже про ВАХ

# import numpy as np
# import matplotlib.pyplot as plt
# from boutdata import collect

# # Параметры импульса и пилообразного сигнала
# amplitude = 100           # амплитуда пилообразного сигнала, В
# impulse_duration = 140e-6 # длительность импульса, с
# sampling_rate = 1000      # частота выборки

# # Убедимся, что num_points всегда больше 0
# num_points = max(1, int(impulse_duration * sampling_rate))

# # Временная шкала и пилообразное напряжение
# time = np.linspace(0, impulse_duration, num_points)
# V_probe = np.linspace(-amplitude, amplitude, num_points)

# # Добавим отладочную печать
# print(f"num_points: {num_points}")
# print(f"impulse_duration: {impulse_duration}")
# print(f"sampling_rate: {sampling_rate}")
# print(f"time shape: {time.shape}")
# print(f"V_probe shape: {V_probe.shape}")

#          # заряд электрона, Кл
# me = 9.11e-31         # масса электрона, кг
# kB = 1.38e-23         # постоянная Больцмана, Дж/К
# probe_area = 6e-5     # площадь зонда, м²


# # Искусственная генерация данных плазмы, если collect не работает
# def generate_plasma_data(num_time_points):
#     """
#     Генерация синтетических данных плазмы
#     """
#     # Базовые параметры плазмы
#     base_density = 1e18  # м^-3
#     base_temperature = 5  # эВ

#     # Создание временных профилей с флуктуациями
#     n_avg = base_density * (1 + 0.1 * np.sin(2 * np.pi * time / impulse_duration))
#     Te_avg = base_temperature * (1 + 0.15 * np.cos(2 * np.pi * time / impulse_duration))

#     return n_avg, Te_avg

# def generate_time_varying_parameters(n_avg, Te_avg, time):
#     """
#     Генерация временных флуктуаций параметров плазмы
#     """
#     # Случайные флуктуации
#     n_fluctuations = np.random.normal(1, 0.1, len(time)) * n_avg
#     Te_fluctuations = np.random.normal(1, 0.15, len(time)) * Te_avg

#     return n_fluctuations, Te_fluctuations

# def current_density(V, n_e, Te):
#     e = 1.6e-19
#     v_th = np.sqrt(kB * Te * e / me)
#     exponent = np.clip(-e * V / (kB * Te * e), -50, 50)
#     return e * n_e * v_th * (1 - np.exp(exponent))

# # Генерация данных плазмы
# n_avg, Te_avg = generate_plasma_data(num_points)

# # Генерация флуктуирующих параметров
# n_e_dynamic, Te_e_dynamic = generate_time_varying_parameters(n_avg, Te_avg, time)

# # Массив для хранения токов
# I_probe_dynamic = np.zeros_like(V_probe)

# # Расчет тока для каждой точки развертки с учетом изменения параметров
# for i in range(num_points):
#     I_probe_dynamic[i] = current_density(V_probe[i], n_e_dynamic[i], Te_e_dynamic[i]) * probe_area

# # Визуализация динамических параметров
# plt.figure(figsize=(15, 10))

# plt.subplot(2, 2, 1)
# plt.plot(time * 1e6, n_e_dynamic, label='Концентрация электронов')
# plt.title('Динамика концентрации')
# plt.xlabel('Время, мкс')
# plt.ylabel('Концентрация, м^-3')
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(time * 1e6, Te_e_dynamic, label='Температура электронов')
# plt.title('Динамика температуры')
# plt.xlabel('Время, мкс')
# plt.ylabel('Температура, эВ')
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(V_probe, I_probe_dynamic)
# plt.title('ВАХ с динамическими параметрами')
# plt.xlabel('Напряжение, В')
# plt.ylabel('Ток, А')

# plt.tight_layout()
# plt.show()
# # Add these print statements before calling reconstruct_parameters
# print("V_probe shape:", V_probe.shape)
# print("I_probe_dynamic shape:", I_probe_dynamic.shape)
# print("V_probe:", V_probe)
# print("I_probe_dynamic:", I_probe_dynamic)
# print("Number of points:", num_points)
# print("Sampling rate:", sampling_rate)
# print("Impulse duration:", impulse_duration)
# # Восстановление параметров
# def reconstruct_parameters(V, I):
#     """
#     Реконструкция параметров плазмы с расширенной диагностикой
#     """
#     # Преобразуем входные данные в numpy массивы с float типом
#     V = np.asarray(V, dtype=float)
#     I = np.asarray(I, dtype=float)

#     # Проверяем на пустоту и корректность размерности
#     if V.size == 0 or I.size == 0:
#         raise ValueError("Input arrays must not be empty")

#     if V.size != I.size:
#         raise ValueError("Input arrays must have the same length")

#     # Обработка тока насыщения
#     try:
#         # Пытаемся найти ток насыщения для отрицательных напряжений
#         neg_voltage_mask = V < 0
#         if np.any(neg_voltage_mask):
#             I_sat_neg = np.max(I[neg_voltage_mask])
#         else:
#             # Если нет отрицательных напряжений, используем максимальный ток
#             I_sat_neg = np.max(I)
#     except Exception as e:
#         print(f"Error finding saturation current: {e}")
#         I_sat_neg = np.max(I)

#     # Оценка температуры
#     try:
#         # Используем логарифмическую аппроксимацию
#         # Фильтруем данные для более надежной оценки
#         valid_mask = (I > 0) & (np.abs(V) > 0)
#         log_current = np.log(np.abs(I[valid_mask]) + 1e-10)
#         voltage_subset = V[valid_mask]

#         if len(voltage_subset) < 2:
#             raise ValueError("Insufficient valid data points")

#         # Линейная регрессия в логарифмических координатах
#         coeffs = np.polyfit(voltage_subset, log_current, 1)
#         T_e_slope = -1 / coeffs[0]
#         e = 1.6e-19
#         T_e_reconstructed = T_e_slope * e / kB

#         # Защита от физически необоснованных значений
#         T_e_reconstructed = max(0.1, min(T_e_reconstructed, 100))

#         # Оценка концентрации
#         n_e_reconstructed = I_sat_neg / (e * probe_area * np.sqrt(2 * np.pi * kB * T_e_reconstructed / me))

#         # Защита от физически необоснованных значений
#         n_e_reconstructed = max(1e16, min(n_e_reconstructed, 1e20))

#         return n_e_reconstructed, T_e_reconstructed

#     except Exception as e:
#         print(f"Error in parameter reconstruction: {e}")
#         # Возвращаем средние значения в случае ошибки
#         return 1e18, 5.0

# # Восстановление и анализ
# n_e_rec, Te_rec = reconstruct_parameters(V_probe, I_probe_dynamic)

# # Статистический анализ
# n_true = np.mean(n_e_dynamic)
# Te_true = np.mean(Te_e_dynamic)

# n_error = np.abs(n_e_rec - n_true) / n_true * 100
# Te_error = np.abs(Te_rec - Te_true) / Te_true * 100
# n_e_dynamic = base_density * (1 + 0.1 * np.sin(2 * np.pi * time / impulse_duration))
# Te_e_dynamic = base_temperature * (1 + 0.15 * np.cos(2 * np.pi * time / impulse_duration))
# print("Анализ восстановления параметров плазмы:")
# print(f"Истинная средняя концентрация: {n_true:.2e} м^-3")
# print(f"Восстановленная концентрация: {n_e_rec:.2e} м^-3")
# print(f"Погрешность концентрации: {n_error:.2f}%")
# print(f"\nИстинная средняя температура: {Te_true:.2f} эВ")
# print(f"Восстановленная температура: {Te_rec:.2f} эВ")
# print(f"Погрешность температуры: {Te_error:.2f}%")


# тоже про ВАХ
# import numpy as np
# import matplotlib.pyplot as plt

# # Постоянные
# e = 1.6e-19  # заряд электрона, Кл
# me = 9.11e-31  # масса электрона, кг
# kB = 1.38e-23  # постоянная Больцмана, Дж/К
# probe_area = 6e-5  # площадь зонда, м²

# # Параметры пилообразного сигнала
# amplitude = 50  # амплитуда напряжения, В
# num_points = 100  # число точек развертки
# V_probe = np.linspace(-amplitude, amplitude, num_points)  # пилообразное напряжение

# # Загрузка данных из расчетов (примерные массивы, заменить на ваши данные)
# # n: концентрация электронов [м^-3]
# # Te: температура электронов [эВ]
# # Эти данные уже должны быть получены из вашего файла
# n = np.mean(collect('n', yguards=True)[:, 2:-2, 2:-2, 0], axis=(1, 2))  # усреднение по пространству
# Te = np.mean(collect('Te', yguards=True)[:, 2:-2, 2:-2, 0], axis=(1, 2))  # усреднение по пространству

# # Подгоняем размер данных к количеству точек развертки
# n = np.interp(np.linspace(0, len(n)-1, num_points), np.arange(len(n)), n)
# Te = np.interp(np.linspace(0, len(Te)-1, num_points), np.arange(len(Te)), Te)

# # Функция для расчета плотности тока
# def current_density(V, n_e, Te_e):
#     v_th = np.sqrt(kB * Te_e * e / me)  # тепловая скорость
#     exponent = np.clip(-e * V / (kB * Te_e * e), -50, 50)  # защита от больших экспонент
#     return e * n_e * v_th * (1 - np.exp(exponent))

# # Расчет ВАХ
# I_probe = np.array([current_density(V, n[i], Te[i]) * probe_area for i, V in enumerate(V_probe)])

# # Построение графика
# plt.figure(figsize=(10, 6))
# plt.plot(V_probe, I_probe, label='ВАХ')
# plt.title('Вольт-амперная характеристика (ВАХ)', fontsize=16)
# plt.xlabel('Напряжение, В', fontsize=14)
# plt.ylabel('Ток, А', fontsize=14)
# plt.grid(True)
# plt.legend()
# plt.show()


# тоже про ВАХ

# import numpy as np
# import matplotlib.pyplot as plt

# # Постоянные
# e = 1.6e-19  # заряд электрона, Кл
# me = 9.11e-31  # масса электрона, кг
# kB = 1.38e-23  # постоянная Больцмана, Дж/К
# probe_area = 6e-5  # площадь зонда, м²

# # Входные данные: V_probe и I_probe
# V_probe = np.linspace(-50, 50, 100)  # напряжение
# I_probe = np.clip(-4e-5 + (1e-6 * np.exp(V_probe / 20)), -4e-5, None)  # синтетический ток с насыщением

# # 1. Определяем насыщенный ток I_sat из отрицательных напряжений
# mask_neg = V_probe < -20  # выбираем область отрицательных напряжений
# I_sat = np.mean(I_probe[mask_neg])  # усреднённый ток насыщения

# # 2. Восстанавливаем концентрацию n_e
# def estimate_concentration(I_sat):
#     v_th = np.sqrt(kB * 300 / me)  # предполагаемая тепловая скорость при 300 K
#     n_e = I_sat / (e * probe_area * v_th)
#     return n_e


# # Расчёт концентрации
# try:
#     n_e = estimate_concentration(I_sat)
#     print(f"Восстановленная концентрация n_e: {n_e:.2e} м^-3")
# except ValueError as e:
#     print(f"Ошибка восстановления концентрации: {e}")

# # Построение графика ВАХ
# plt.figure(figsize=(10, 6))
# plt.plot(V_probe, I_probe, label='ВАХ')
# plt.axhline(I_sat, color='red', linestyle='--', label=f'I_sat = {I_sat:.2e} A')
# plt.title('Вольт-Амперная характеристика', fontsize=16)
# plt.xlabel('Напряжение, В', fontsize=14)
# plt.ylabel('Ток, А', fontsize=14)
# plt.grid(True)
# plt.legend()
# plt.show()


# для ВАХ который так и не построился нормально.

# # Постоянные
# e = 1.6e-19  # заряд электрона, Кл
# me = 9.11e-31  # масса электрона, кг
# kB = 1.38e-23  # постоянная Больцмана, Дж/К
# probe_area = 6e-5  # площадь зонда, м²

# # Параметры пилообразного сигнала
# amplitude = 50  # амплитуда напряжения, В
# num_points = 100  # число точек развертки
# V_probe = np.linspace(-amplitude, amplitude, num_points)  # пилообразное напряжение

# # Загрузка синтетических данных
# n = np.linspace(1e16, 1e17, num_points)  # концентрация электронов [м^-3]
# Te = np.linspace(2, 5, num_points)  # температура электронов [эВ]

# # Функция для расчета плотности тока
# def current_density(V, n_e, Te_e):
#     v_th = np.sqrt(kB * Te_e * e / me)  # тепловая скорость
#     exponent = np.clip(-e * V / (kB * Te_e * e), -50, 50)  # защита от больших экспонент
#     return e * n_e * v_th * (1 - np.exp(exponent))

# # Расчет ВАХ
# I_probe = np.array([current_density(V, n[i], Te[i]) * probe_area for i, V in enumerate(V_probe)])

# # Определяем ток насыщения I_sat из отрицательных напряжений
# mask_neg = V_probe < -10  # область отрицательных напряжений
# I_sat = np.mean(I_probe[mask_neg])  # усреднённый ток насыщения

# # Выделяем экспоненциальный участок для логарифмирования
# mask_exp = (V_probe > 5) & (I_probe > I_sat * 1.01)
# V_exp = V_probe[mask_exp]
# I_exp = I_probe[mask_exp]

# # Логарифмируем разность тока и тока насыщения
# log_I = np.log(I_exp - I_sat)

# # Линейная аппроксимация log(I - I_sat) от V
# coeffs = np.polyfit(V_exp, log_I, 1)
# slope = coeffs[0]  # наклон прямой

# # Восстановление температуры электронов
# Te_reconstructed = -e / (kB * slope)
# print(f"Восстановленная температура электронов: {Te_reconstructed:.2f} эВ")

# # Построение графиков
# plt.figure(figsize=(10, 6))
# plt.plot(V_probe, I_probe, label='ВАХ', color='blue')
# plt.axhline(I_sat, color='red', linestyle='--', label=f'I_sat = {I_sat:.2e} A')
# plt.plot(V_exp, I_exp, 'go', label='Экспоненциальный участок')
# plt.title('Вольт-Амперная характеристика (ВАХ)', fontsize=16)
# plt.xlabel('Напряжение, В', fontsize=14)
# plt.ylabel('Ток, А', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.show()

# # Логарифмический график
# plt.figure(figsize=(10, 6))
# plt.plot(V_exp, log_I, 'o', label='ln(I - I_sat)')
# plt.plot(V_exp, coeffs[0] * V_exp + coeffs[1], label=f'Аппроксимация (slope={slope:.3f})', color='orange')
# plt.title('Логарифмический график тока на экспоненциальном участке', fontsize=16)
# plt.xlabel('Напряжение, В', fontsize=14)
# plt.ylabel('ln(I - I_sat)', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.show()


# плотность плазмы по радиусу на растояний
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# # from your_module import collect  # Импорт вашей функции для загрузки данных

# # Генерация списка папок
# base_path = r'C:\Users\auezo\Desktop\fromcluster'
# data_dirs = [os.path.join(base_path, f'Test{i}') for i in range(1, 63)]  # Список от Test1 до Test62

# # Список для проблемных папок
# failed_dirs = []

# def plot_density_by_radius(data_dir, title, nframes, z_positions, Rcell, Lcell, cell_R, n_factor=1.0):
#     """
#     Построить графики распределения плотности плазмы по радиусу на заданных расстояниях z для указанной папки.

#     Parameters:
#     - data_dir: str, путь к папке с данными
#     - title: str, заголовок графика
#     - остальные параметры аналогичны
#     """
#     os.chdir(data_dir)  # Переход в папку с данными

#     # Загружаем плотность плазмы
#     n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Функция collect подгружает данные

#     # Проверка максимального доступного индекса
#     max_index = n.shape[0] - 1  # Максимальный индекс по оси времени
#     if nframes > max_index:
#         print(f"Предупреждение: nframes={nframes} превышает доступный размер {max_index + 1}. Используется последний кадр.")
#         nframes = max_index  # Выбираем последний доступный кадр

#     n_140us = n[nframes] * n_factor  # Берем кадр и масштабируем

#     # Координаты
#     nx, ny = n_140us.shape
#     r = np.linspace(0, Rcell, nx)  # Радиальная координата
#     z = np.linspace(0, Lcell, ny)  # Продольная координата

#     # Построение графиков
#     plt.figure(figsize=(10, 6))

#     for z_pos in z_positions:
#         z_index = np.argmin(np.abs(z - z_pos))  # Ищем ближайший индекс
#         plt.plot(r, n_140us[:, z_index], label=f'z = {z_pos:.1f} см')

#     # Добавляем границу накопительной ячейки
#     plt.axvline(x=cell_R, color='red', linestyle='--', label='Граница накопительной ячейки')

#     # Настройка графика
#     plt.title(title, fontsize=16, fontweight='bold')  # Подписываем график названием набора данных
#     plt.xlabel('Радиус r, мм', fontsize=14, fontweight='bold')
#     plt.ylabel('Плотность плазмы, × $10^{19}$ част/м³', fontsize=14, fontweight='bold')
#     plt.xlim(0, Rcell)  # Устанавливаем начало оси X с нуля
#     plt.legend(fontsize=16)
#     plt.grid(False)
#     plt.tick_params(axis='both', which='major', labelsize=14)
#     plt.tight_layout()
#     plt.show()

# # Цикл по сгенерированным папкам
# for data_dir in data_dirs:
#     folder_name = os.path.basename(data_dir)  # Имя папки, например Test22
#     try:
#         plot_density_by_radius(
#             data_dir=data_dir,
#             title=f'График для {folder_name}',  # Подпись графика
#             nframes=100,  # Пример
#             z_positions=[1.0, 2.0, 3.0],  # Пример координат
#             Rcell=10.0,  # Пример радиуса
#             Lcell=20.0,  # Пример длины
#             cell_R=5.0,  # Пример границы ячейки
#             n_factor=1.0  # Пример масштаба
#         )
#     except (OSError, FileNotFoundError, IndexError) as e:
#         print(f"Ошибка с папкой {folder_name}: {e}")
#         failed_dirs.append(folder_name)

# # Вывод всех проблемных папок
# if failed_dirs:
#     print("\nПроблемы возникли с папками:")
#     for folder in failed_dirs:
#         print(f"- {folder}")
# else:
#     print("\nВсе папки обработаны успешно!")
