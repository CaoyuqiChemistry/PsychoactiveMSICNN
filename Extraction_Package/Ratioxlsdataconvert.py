from progressbar import *
import numpy as np
from scipy.interpolate import griddata
import xlwt,xlrd
import matplotlib
import cv2
matplotlib.use('Qt5Agg')

def points_confirm(x,y,point_data):
    c=np.where(point_data[:,0]==x)[0]
    m=np.where(point_data[c,1]==y)[0]
    if len(m)==0:
        return -1
    else:
        return c[m[0]]

def MSI_xls_data(MSI_dir,index,size_x,size_y):
    data = xlrd.open_workbook(MSI_dir)
    table = data.sheets()[0]
    nrows = table.nrows

    add_n = 7
    subtract_points = np.array([[table.cell_value(1, 1), table.cell_value(1, 2)]])
    subtract_intensity = np.array([table.cell_value(1, 3*index+add_n)])

    for i in range(2, nrows):
        subtract_points = np.append(subtract_points, [[table.cell_value(i, 1), table.cell_value(i, 2)]], axis=0)
        subtract_intensity = np.append(subtract_intensity, [table.cell_value(i, 3*index+add_n)], axis=0)

    total_points = []
    total_intensity = []

    blank_intensity = 0

    xmin, xmax = int(subtract_points[:, 0].min()), int(subtract_points[:, 0].max())

    ymin, ymax = int(subtract_points[:, 1].min()), int(subtract_points[:, 1].max())

    for x in range(xmin - 5, xmax + 5):
        for y in range(ymin - 5, ymax + 5):
            index = points_confirm(x, y, subtract_points)
            total_points.append([x, y])
            if index == -1:
                total_intensity.append(blank_intensity)
            else:
                total_intensity.append(subtract_intensity[index])
    total_points = np.array(total_points)
    total_intensity = np.array(total_intensity)
    grid_x, grid_y = np.mgrid[(xmin - 5):(xmax + 4):size_x * 1j,
                     (ymin - 5):(ymax + 4):size_y * 1j]
    grid_z0 = griddata(total_points, total_intensity, (grid_x, grid_y), method='linear')

    return grid_z0

def total_MSI_data(MSI_dir,file_path,size_x,size_y):
    final_grid = []
    for index in range(0,15):
        tmp_grid = MSI_xls_data(MSI_dir,index,size_x,size_y)
        tmp_grid = tmp_grid / tmp_grid.max()
        final_grid.append(tmp_grid)
    final_grid = cv2.merge(final_grid)
    np.save(file_path,final_grid)
    print('Successfully Saved!')
    return final_grid

def MSI_xls_data_New(MSI_dir,index,size_x,size_y,percent=(0,0,0,0)):
    data = xlrd.open_workbook(MSI_dir)
    table = data.sheets()[0]
    nrows = table.nrows

    add_n = 7
    subtract_points = np.array([[table.cell_value(1, 1), table.cell_value(1, 2)]])
    subtract_intensity = np.array([table.cell_value(1, 3*index+add_n)])

    for i in range(2, nrows):
        subtract_points = np.append(subtract_points, [[table.cell_value(i, 1), table.cell_value(i, 2)]], axis=0)
        subtract_intensity = np.append(subtract_intensity, [table.cell_value(i, 3*index+add_n)], axis=0)

    total_points = []
    total_intensity = []

    blank_intensity = 0

    xmin, xmax = int(subtract_points[:, 0].min()), int(subtract_points[:, 0].max())

    ymin, ymax = int(subtract_points[:, 1].min()), int(subtract_points[:, 1].max())

    x_left, x_right = int(xmin*(1-percent[0])), int(xmax*(1+percent[1]))
    y_left, y_right = int(ymin*(1-percent[2])), int(ymax*(1-percent[3]))
    for x in range(x_left-5, x_right+5):
        for y in range(y_left-5, y_right+5):
            index = points_confirm(x, y, subtract_points)
            total_points.append([x, y])
            if index == -1:
                total_intensity.append(blank_intensity)
            else:
                total_intensity.append(subtract_intensity[index])
    total_points = np.array(total_points)
    total_intensity = np.array(total_intensity)
    grid_x, grid_y = np.mgrid[(x_left-5):(x_right+4):(size_x * 1j),(y_left-5): (y_right+4):(size_y * 1j)]
    grid_z0 = griddata(total_points, total_intensity, (grid_x, grid_y), method='linear')

    return grid_z0
