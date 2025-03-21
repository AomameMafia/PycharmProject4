#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для построения графиков плотности плазмы.
"""
import os
os.add_dll_directory(r"C:\Program Files\netCDF 4.9.3\bin")
os.add_dll_directory(r"C:\Program Files\HDF_Group\HDF5\1.14.6\bin")





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для построения графиков плотности плазмы.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, ticker
from boutdata import collect
from constant import Rcell, Lcell, cell_R, cell_r, cell_L, Nx
from data_loader import load_plasma_data

# --- Участок 1: График плотности по радиусу ---
def plot_density_by_radius(data_dir, nframes, z_positions, n_factor=1.0):
    """
    Построение линейного графика плотности плазмы в зависимости от радиуса на заданных z.
    - data_dir: путь к папке с данными (BOUT.dmp.*.nc).
    - nframes: номер кадра времени (например, 50).
    - z_positions: список координат z (например, [7.0, 14.0]) в см.
    - n_factor: множитель для масштабирования плотности (по умолчанию 1.0).
    """
    os.chdir(data_dir)  # Переключаемся в папку с данными
    n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Загружаем плотность из файлов
    n_frame = n[nframes] * n_factor  # Выбираем кадр и масштабируем
    nx, ny = n_frame.shape  # Размеры массива (радиус, z)
    r = np.linspace(0, Rcell, nx)  # Координаты радиуса в мм
    z = np.linspace(0, Lcell, ny)  # Координаты z в см
    plt.figure(figsize=(10, 6))  # Создаём фигуру размером 10x6 дюймов
    for z_pos in z_positions:  # Цикл по заданным z
        z_index = np.argmin(np.abs(z - z_pos))  # Находим индекс ближайшего z
        plt.plot(r, n_frame[:, z_index], label=f'z = {z_pos:.1f} см')  # График плотности по радиусу
    plt.axvline(x=cell_R, color='red', linestyle='--', label='Граница накопительной ячейки')  # Линия границы ячейки
    plt.xlabel('Радиус r, мм', fontsize=14, fontweight='bold')  # Подпись оси X
    plt.ylabel('Плотность плазмы, × $10^{13}$ част/м³', fontsize=14, fontweight='bold')  # Подпись оси Y
    plt.legend(fontsize=12)  # Легенда
    plt.tight_layout()  # Оптимизация расположения
    plt.show()  # Показываем график
#
# # --- Участок 2: 2D-график плотности в логарифмической шкале ---
# def plot_density_2d(n, nframes, time, minvar, maxvar, figsizex=18.0, figsizey=8.0):
#     """
#     Построение 2D-контурного графика плотности в логарифмической шкале для нескольких моментов времени.
#     - n: массив плотности (уже загруженный).
#     - nframes: список номеров кадров (например, [12, 25, 50, 200]).
#     - time: массив времени в мкс.
#     - minvar, maxvar: минимальное и максимальное значения для цветовой шкалы.
#     - figsizex, figsizey: размеры фигуры в см.
#     """
#     nt, nx, ny = n.shape  # Размеры массива (время, радиус, z)
#     nf = len(nframes)  # Количество кадров
#     x = np.linspace(0., Rcell, nx)  # Координаты радиуса в см
#     y = np.linspace(0., Lcell, ny)  # Координаты z в см
#     Y, X = np.meshgrid(y, x)  # Сетка для 2D-графика
#     f = plt.figure(figsize=(figsizex / 2.54, figsizey / 2.54))  # Фигура в дюймах (перевод из см)
#     wr = np.ones(nf)  # Равные ширины панелей
#     gs = gridspec.GridSpec(1, nf, width_ratios=wr)  # Сетка для нескольких графиков
#     colorlev = np.linspace(minvar, maxvar, 100)  # Уровни цвета
#     cmap = plt.cm.hot_r  # Цветовая карта (красный градиент)
#     for i in range(nf):  # Цикл по кадрам
#         idt = nframes[i]  # Индекс кадра
#         ax = f.add_subplot(gs[0, i])  # Добавляем панель
#         if i == 0:  # Подпись оси Y только на первом графике
#             ax.set_ylabel(r'$z,\ \mathrm{cm}$', fontsize=28)
#         else:
#             plt.setp(ax.get_yticklabels(), visible=False)  # Убираем метки Y на остальных
#             ax.tick_params(axis='y', which='both', bottom=False, left=False)  # Убираем тики
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # Основные метки X
#         ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))  # Дополнительные метки X
#         ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))  # Дополнительные метки Y
#         cs = ax.contourf(X, Y, n[idt, :, :], levels=colorlev, cmap=cmap, extend='both')  # Контурный график
#         ax.plot([cell_r, cell_R], [cell_L, cell_L], 'b', linewidth=2.0)  # Нижняя граница ячейки
#         ax.plot([cell_R, cell_R], [cell_L, Lcell], 'b', linewidth=2.0)  # Правая граница ячейки
#         ax.set_title('%d' % int(round(time[nframes[i]])), fontsize=20, pad=20)  # Заголовок с временем
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # Основные метки X на последнем графике
#     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))  # Дополнительные метки X
#     cbar_ax = f.add_axes([0.9, 0.2, 0.015, 0.675])  # Место для цветовой шкалы
#     cax = f.colorbar(cs, cax=cbar_ax, ticks=np.linspace(minvar, maxvar, 5), orientation='vertical', extend='both')  # Шкала
#     cax.ax.set_yticklabels(['%d' % t for t in np.linspace(minvar, maxvar, 5)])  # Метки на шкале
#     cax.ax.tick_params(labelsize=10)  # Размер меток
#     f.text(0.5, 0.04, r'$r,\ \mathrm{cm}$', ha='center', fontsize=28)  # Подпись оси X
#     f.suptitle(r'$t,\ \mathrm{\mu{s}}$:', fontsize=24, x=0.05, y=0.98)  # Главный заголовок
#     f.subplots_adjust(wspace=0.3, left=0.1, bottom=0.15, right=0.875, top=0.9)  # Настройка отступов
#     plt.show()  # Показываем график
#
# # --- Участок 3: 2D-график плотности в линейной шкале ---
# def plot_density(data_dir, nframes, minvar, maxvar, figsizex=18.0, figsizey=8.0):
#     """
#     Построение 2D-контурного графика плотности в линейной шкале для нескольких моментов времени.
#     - data_dir: путь к папке с данными.
#     - nframes: список номеров кадров (например, [13, 25, 50, 200]).
#     - minvar, maxvar: минимальное и максимальное значения для цветовой шкалы.
#     - figsizex, figsizey: размеры фигуры в см.
#     """
#     os.chdir(data_dir)  # Переключаемся в папку с данными
#     n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Загружаем плотность
#     nt, nx, ny = n.shape  # Размеры массива
#     nf = len(nframes)  # Количество кадров
#     x = np.linspace(0., Rcell, nx)  # Координаты радиуса в см
#     y = np.linspace(0., Lcell, ny)  # Координаты z в см
#     Y, X = np.meshgrid(y, x)  # Сетка для 2D-графика
#     f = plt.figure(figsize=(figsizex/2.54, figsizey/2.54))  # Фигура в дюймах
#     wr = np.ones(nf)  # Равные ширины панелей
#     gs = gridspec.GridSpec(1, nf, width_ratios=wr)  # Сетка
#     colorlev = np.linspace(minvar, maxvar, 100)  # Уровни цвета
#     cmap = plt.cm.hot_r  # Цветовая карта
#     for i in range(nf):  # Цикл по кадрам
#         idt = nframes[i]  # Индекс кадра
#         ax = f.add_subplot(gs[0, i])  # Добавляем панель
#         if i == 0:  # Подпись оси Y только на первом графике
#             ax.set_ylabel(r'$z,\ \mathrm{cm}$', fontsize=28)
#         else:
#             plt.setp(ax.get_yticklabels(), visible=False)  # Убираем метки Y
#             ax.tick_params(axis='y', which='both', bottom=False, left=False)  # Убираем тики
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # Основные метки X
#         ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))  # Дополнительные метки X
#         ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))  # Дополнительные метки Y
#         cs = ax.contourf(X, Y, n[idt, :, :], levels=colorlev, cmap=cmap, extend='both')  # Контурный график
#         ax.plot([cell_r, cell_R], [cell_L, cell_L], 'b', linewidth=2.0)  # Нижняя граница ячейки
#         ax.plot([cell_R, cell_R], [cell_L, Lcell], 'b', linewidth=2.0)  # Правая граница ячейки
#         ax.set_xlabel(r'$r,\ \mathrm{cm}$', fontsize=28)  # Подпись оси X
#         ax.set_title('$t\ =\ {%s}\ \mathrm{\mu{s}}$' % str(round(time[nframes[i]], 2)), fontsize=12)  # Заголовок с временем
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # Основные метки X
#     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))  # Дополнительные метки X
#     cbar_ax = f.add_axes([0.9, 0.2, 0.015, 0.675])  # Место для цветовой шкалы
#     cax = f.colorbar(cs, cax=cbar_ax, ticks=np.linspace(minvar, maxvar, 5), orientation='vertical', extend='both')  # Шкала
#     cax.ax.tick_params(labelsize=10)  # Размер меток
#     f.subplots_adjust(wspace=0.0, left=0.1, bottom=0.1, right=0.875, top=0.95)  # Настройка отступов
#     plt.show()  # Показываем график
#
# # --- Участок 4: График средней плотности ---
# def plot_avg_density(data_dir, minvar, maxvar, figsizex=18.0, figsizey=8.0):
#     """
#     Построение 2D-контурного графика средней плотности по времени в нормализованных координатах.
#     - data_dir: путь к папке с данными.
#     - minvar, maxvar: минимальное и максимальное значения для цветовой шкалы.
#     - figsizex, figsizey: размеры фигуры в см.
#     """
#     os.chdir(data_dir)  # Переключаемся в папку с данными
#     n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Загружаем плотность
#     nt, nx, ny = n.shape  # Размеры массива
#     x = np.linspace(0., 1., nx)  # Нормированный радиус (r/R)
#     y = np.linspace(0., 1., ny)  # Нормированная координата z (z/L)
#     Y, X = np.meshgrid(y, x)  # Сетка для 2D-графика
#     f = plt.figure(figsize=(figsizex/2.54, figsizey/2.54))  # Фигура в дюймах
#     plt.xlabel(r'$r/R$', fontsize=12)  # Подпись оси X
#     plt.ylabel(r'$z/L$', fontsize=12)  # Подпись оси Y
#     colorlev = np.linspace(minvar, maxvar, 100)  # Уровни цвета
#     cmap = plt.cm.hot_r  # Цветовая карта
#     cs = plt.contourf(X, Y, np.mean(n[10:,:,:], axis=0), levels=colorlev, cmap=cmap, extend='both')  # Средняя плотность
#     ax = f.gca()  # Текущая ось
#     ax.xaxis.set_major_locator(ticker.FixedLocator([0., 0.25, 0.5, 0.75]))  # Основные метки X
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # Основные метки Y
#     ax.xaxis.set_major_locator(ticker.FixedLocator([0., 0.25, 0.5, 0.75, 1.0]))  # Уточнённые метки X
#     cbar_ax = f.add_axes([0.85, 0.13, 0.015, 0.82])  # Место для цветовой шкалы
#     cax = f.colorbar(cs, cax=cbar_ax, ticks=np.arange(minvar, maxvar, 0.5), orientation='vertical', extend='both')  # Шкала
#     cax.ax.tick_params(labelsize=10)  # Размер меток
#     f.subplots_adjust(wspace=0.0, left=0.19, bottom=0.13, right=0.8, top=0.95)  # Настройка отступов
#     plt.show()  # Показываем график
#
# # --- Участок 5: График плотности по диаметру ---
# def plot_density_by_diameter(data_dir, nframes, z_positions, n_factor=1.0):
#     """
#     Построение линейного графика плотности вдоль диаметра на заданных z.
#     - data_dir: путь к папке с данными.
#     - nframes: номер кадра времени (например, 50).
#     - z_positions: список координат z (например, [7.0, 14.0]) в см.
#     - n_factor: множитель для масштабирования плотности (по умолчанию 1.0).
#     """
#     os.chdir(data_dir)  # Переключаемся в папку с данными
#     n = collect('n', yguards=True)[:, 2:-2, 2:-2, 0]  # Загружаем плотность
#     n_140us = n[nframes] * n_factor  # Выбираем кадр и масштабируем
#     nx, ny = n_140us.shape  # Размеры массива
#     d = np.linspace(-Rcell, Rcell, 2 * nx)  # Координаты диаметра в см (симметрично)
#     z = np.linspace(0, Lcell, ny)  # Координаты z в см
#     plt.figure(figsize=(12, 8))  # Фигура размером 12x8 дюймов
#     for z_pos in z_positions:  # Цикл по заданным z
#         z_index = np.argmin(np.abs(z - z_pos))  # Индекс ближайшего z
#         profile = np.concatenate([n_140us[::-1, z_index], n_140us[:, z_index]])  # Симметричный профиль
#         plt.plot(d, profile, label=f'z = {z_pos:.1f} см')  # График плотности по диаметру
#     plt.axvline(x=-cell_R, color='red', linestyle='--', label='Левая граница накопительной ячейки')  # Левая граница
#     plt.axvline(x=cell_R, color='red', linestyle='--', label='Правая граница накопительной ячейки')  # Правая граница
#     plt.title('Распределение плотности плазмы по диаметру', fontsize=18, fontweight='bold')  # Заголовок
#     plt.xlabel('Диаметр d, см', fontsize=16, fontweight='bold')  # Подпись оси X
#     plt.ylabel('Плотность плазмы, × $10^{13}$ част/м³', fontsize=16, fontweight='bold')  # Подпись оси Y
#     plt.xticks(fontsize=14)  # Размер меток X
#     plt.yticks(fontsize=14)  # Размер меток Y
#     plt.legend(fontsize=14)  # Легенда
#     plt.grid(True, linestyle='--', alpha=0.7)  # Сетка
#     plt.tight_layout()  # Оптимизация расположения
#     plt.show()  # Показываем график

# --- Основной блок: Запуск всех графиков ---
if __name__ == "__main__":
    # Укажите путь к вашей папке с данными
    data_dir = r"D:\fromcluster\Test9"  # Замените на ваш путь
    try:
        # Загружаем данные один раз для всех графиков
        n, _, _, _, time = load_plasma_data(data_dir)
        # Вызываем функции для построения графиков
        plot_density_by_radius(data_dir, 5, [7.0, 14.0], n_factor=4.2e16)  # График плотности по радиусу
        # plot_density_2d(n, (12, 25, 50, 200), time, 15, 19.5)  # 2D-график в логарифмической шкале
        # plot_density(data_dir, (13, 25, 50, 200), 0, 1.25, 16., 10.)  # 2D-график в линейной шкале
        # plot_avg_density(data_dir, 16.5, 18.5, 8., 10.)  # График средней плотности
        # plot_density_by_diameter(data_dir, 50, [7.0, 14.0], n_factor=4.2e16)  # График плотности по диаметру
    except Exception as e:
        print(f"Ошибка при загрузке данных или построении графиков: {e}")