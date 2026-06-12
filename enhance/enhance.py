main.py
import math
import os
import re
import struct
import sys
import time
import random

import numpy as np
from pathlib import Path
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from visualize3Dpgm import visualize_pointcloud, load_image_as_array, polar_to_pointcloud
from models.m20.generator import *

class WeatherPGMGeneraor:
    def __init__(self, generator, x_ind = 0, y_ind = 1, z_ind = 2):
        self.x_ind = x_ind
        self.y_ind = y_ind
        self.z_ind = z_ind
        self.v_fov = 40.0 * math.pi / 180.0
        self.v_fov_h = 20.0 * math.pi / 180.0
        self.v_fov_l = self.v_fov_h - self.v_fov
        self.h_fov = math.pi * 2.0
        self.far_plane = 100.0
        # Image Parameters
        self.width = 1024.0
        self.height = 32.0

        self.generator = generator

    def get_centered_yaw(self, x, y):
        if y < 0:
            if x < 0:
                return math.atan2(y, x) + 3 * math.pi / 2.0
            return math.atan2(y, x) - math.pi / 2.0
        return math.atan2(y, x) - math.pi / 2.0

    def get_label(self, u):
        id = (u >> 16) & 0xFFFF
        label = u & 0xFFFF
        return id, label, self.get_color_from_class(label)

    def get_color_from_class(self, id_class):
        """ Hardcoddato da https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml """
        color_map = {
            0: [0, 0, 0],
            1: [0, 0, 255],
            10: [245, 150, 100],
            11: [245, 230, 100],
            13: [250, 80, 100],
            15: [150, 60, 30],
            16: [255, 0, 0],
            18: [180, 30, 80],
            20: [255, 0, 0],
            30: [30, 30, 255],
            31: [200, 40, 255],
            32: [90, 30, 150],
            40: [255, 0, 255],
            44: [255, 150, 255],
            48: [75, 0, 75],
            49: [75, 0, 175],
            50: [0, 200, 255],
            51: [50, 120, 255],
            52: [0, 150, 255],
            60: [170, 255, 150],
            70: [0, 175, 0],
            71: [0, 60, 135],
            72: [80, 240, 150],
            80: [150, 240, 255],
            81: [0, 0, 255],
            99: [255, 255, 50],
            252: [245, 150, 100],
            256: [255, 0, 0],
            253: [200, 40, 255],
            254: [30, 30, 255],
            255: [90, 30, 150],
            257: [250, 80, 100],
            258: [180, 30, 80],
            259: [255, 0, 0]
        }
        p = tuple(color_map.get(id_class, [0, 0, 0]))
        return p

    def get_label_distribution(self, points_dir, label_dir):
        class_distr = np.zeros((int(self.height), int(self.width), 3), dtype=np.float32)
        label_files = sorted(os.listdir(label_dir))
        points_files = sorted(os.listdir(points_dir))
        num_of_pointclouds = len(points_files)
        label_color = self.get_color_from_class(50)

        # Assicurati che i file abbiano corrispondenza (ad esempio, 1.label corrisponde a 1.bin)
        i = 1
        for label_file, points_file in zip(label_files, points_files):
            print(f'{i} / {num_of_pointclouds}')
            i += 1
            label = os.path.join(label_dir, label_file)
            points = os.path.join(points_dir, points_file)
            points = np.fromfile(points, dtype=np.float32)
            label = np.fromfile(label, dtype=np.uint32)
            points = points.reshape((-1, 4))  # [n_punti, x,y,z,i]

            _, pgm_label = self.bin_to_pgm(points, label, as_numpy_array=True)
            mask = np.all(pgm_label == label_color, axis=-1)
            pgm_label = np.zeros_like(pgm_label)
            pgm_label[mask] = label_color
            class_distr += (pgm_label/255.0)/num_of_pointclouds
            #if i == 2: break

        #return _, 1.0 - (class_distr - 1.0)*(class_distr - 1.0)*(class_distr - 1.0)*(class_distr - 1.0)
        #return _, -class_distr*class_distr + 2.0 * class_distr
        return _,class_distr

    def get_x_y_z(self, i, points): # Change this based on dataset used (see DatasetNotes.md)
        return  (points[i][self.x_ind],
                points[i][self.y_ind],
                points[i][self.z_ind])
    
    def get_nv(self, pitch)
        return self.get_nv_from_pitch(pitch, [])            # use pitch
        # return self.get_nv_from_list_distribution(pitch)  # use distribution

    def bin_to_pgm(self, points, label, as_numpy_array = False):
        highest_pitch = -90.0
        lowest_pitch = 90.0
        max_yaw = -200.0
        min_yaw = 200.0
        num_points = max(points.shape[0], points.shape[1])

        if not as_numpy_array:
            pgm = Image.new('F', (int(self.width), int(self.height)), color=255)
            pgm_label = Image.new('RGB', (int(self.width), int(self.height)), color=(255,255,0))
            original_zeros = np.zeros((int(self.height),(int(self.width))))
        else:
            pgm = np.ones((int(self.height),(int(self.width))))
            pgm_label = np.zeros((int(self.height),(int(self.width)), 3), dtype=int)
            original_zeros = np.zeros((int(self.height),(int(self.width))))

        if not as_numpy_array:
            pgm_draw = ImageDraw.Draw(pgm)
            label_draw = ImageDraw.Draw(pgm_label)

        # Calibration phase
        max_d = 0
        for i in range(num_points):
            x,y,z = get_x_y_z(i,points)

            d = pow(x * x + y * y + z * z, 0.5)
            pitch = 0.0

            if d > 0.00001:
                pitch = math.asin(z / d)
            if d>max_d: max_d = d

            yaw = self.get_centered_yaw(x, y)

            if highest_pitch < pitch:
                highest_pitch = pitch
            if lowest_pitch > pitch:
                lowest_pitch = pitch
            if yaw > max_yaw:
                max_yaw = yaw
            if yaw < min_yaw:
                min_yaw = yaw

            r_v_fov = (highest_pitch - lowest_pitch) * 180.0 / math.pi
            r_v_fov_h = highest_pitch * 180.0 / math.pi
            r_h_fov = max(max_yaw, abs(min_yaw)) * 2.0

            r_v_fov = self.ceil_at_second_decimal(r_v_fov)
            r_v_fov_h = self.ceil_at_second_decimal(r_v_fov_h)
            r_h_fov = self.ceil_at_first_decimal(r_h_fov)

            self.v_fov = r_v_fov * math.pi / 180.0
            self.v_fov_h = r_v_fov_h * math.pi / 180.0
            self.v_fov_l = self.v_fov_h - self.v_fov
            self.h_fov = r_h_fov

        print(f"MAX D: {max_d}")
        # PGM Generation phase
        for i in range(num_points):
            x,y,z = get_x_y_z(i,points)

            d = pow(x * x + y * y + z * z, 0.5)
            pitch = 0.0

            if d > 0.0001:
                pitch = math.asin(z / d)
            if d > self.far_plane:
                d = self.far_plane

            if pitch > self.v_fov_h or pitch < self.v_fov_l:
                continue

            yaw = self.get_centered_yaw(x, y)
            nu = (yaw - self.h_fov / 2.0) / (- self.h_fov)

            
            nv = self.get_nv(pitch)

            if nu > 1.0 or nu < 0.0:
                continue
            if nv > 1.0 or nv < 0.0:
                continue

            u = nu * self.width
            v = nv * self.height

            args = {"max_distance": self.far_plane, "poly_degree": 4.0}
            pixel_intensity = self.get_pixel_value_from_depth(d, mode='linear', args=args)
            if label is not None:
                _,_,label_color = self.get_label(label[i])

            if not as_numpy_array:
                pgm_draw.point((u, v), pixel_intensity)
                original_zeros[int(v)][int(u)] = 1.0
                #pgm_draw.point((u,v), points[i][3] * 255)  # <--- SCOMMENTA PER SALVARE L'INTENSITA'
                if label is not None:
                    label_draw.point((u, v), label_color)
            else:
                pgm[int(v), int(u)] = pixel_intensity/255.0
                original_zeros[int(v)][int(u)] = 1.0
                if label is not None:
                    pgm_label[int(v), int(u),:] = label_color

        # Some datasets (cadcd) have white rows (problem with projection).
        # interpolated the selected row with the one above and below.
        # otherwise set selected_row = - 1
        #selected_row = -1
        #if selected_row >= 0:
        #    for i in range(int(self.width)):
        #        a = pgm.getpixel((i, selected_row + 1))
        #        b = pgm.getpixel((i, selected_row - 1))
        #        pgm_draw.point((i, selected_row), (a + b) / 2.0)

        if not as_numpy_array:
            pgm = pgm.convert('L')

        return pgm, pgm_label, original_zeros

    def ceil_at_second_decimal(self, n):
        num = n
        num *= 100.0
        num += 1
        num = math.floor(num)
        num /= 100.0
        return num

    def ceil_at_first_decimal(self, n):
        num = n
        num *= 10.0
        num += 1
        num = math.floor(num)
        num /= 10.0
        return num

    def get_nv_from_pitch(self, pitch, args, mode='linear'):
        nv = 0
        if mode == 'linear':
            return 1.0 - (pitch - self.v_fov_l) / (self.v_fov_h - self.v_fov_l)
        if mode == 'sigmoid':
            o = args['offset']
            nv_spread = args['nv_spread']
            nv = 1.0 / (1 + math.e ** (nv_spread * (pitch - o)))
        if mode == 'poly':
            o = args['offset']
            nv_spread = args['nv_spread']
            value = (self.v_fov_h + self.v_fov_l - 2.0 * pitch - o) / (self.v_fov_h - self.v_fov_l - o)
            if value > 0:
                nv = 0.5 + 0.5 * value ** (1.0 / nv_spread)
            else:
                value = (-self.v_fov_h - self.v_fov_l + 2.0 * pitch + o) / (self.v_fov_h - self.v_fov_l + o)
                nv = 0.5 - 0.5 * value ** (1.0 / nv_spread)
        return nv

    def get_nv_from_list_distribution(self, pitch, lst=None):
        dist = lst
        if dist is None:
            # Velodyne Ultra Puck (VLP - 32C) Elevations distribution
            dist = [15.0, 10.333, 7.0, 4.667, 3.333, 2.333, 1.667, 1.333,
                    1.0, 0.667, 0.333, 0, -0.333, -0.667, -0.843, -1.0,
                    -1.333, -1.667, -2.0, -2.333, -2.667, -3.0, -3.333, -3.667,
                    -4.0, -4.667, -5.333, -6.148, -7.254, -11.31, -15.639, -25.0
                    ]
        # Return the closest index of the list (basically the ring of the point)
        return min(range(len(dist)), key=lambda i: abs(dist[i] - (pitch * 180.0/math.pi))) / self.height

    def get_pixel_value_from_depth(self, depth, args, mode='linear'):
        assert args['max_distance'] > 0
        assert mode in ['linear', 'poly', 'log'], "Mode must be 'linear', 'poly', or 'log'"
        if mode == 'poly':
            assert args['poly_degree'] > 1

        d = args['max_distance']
        p = args['poly_degree']

        if mode == 'linear':
            return (depth * 255.0) / d
        if mode == 'poly':
            return -math.pow((1 - depth / d), p) * 255.0 + 255.0
        if mode == 'log':
            return math.log(depth + 1) * 255.0 / math.log(d)
        return None

    def get_mask_from_pgm(self, pgm, target_weather, target_intensity):
        _pgm = torch.tensor(pgm, dtype=torch.float32)
        _pgm = torch.zeros(16,1,32,1024)
        _target_label = torch.zeros((16,4))
        _pgm[0] = torch.tensor(pgm)
        _target_label[0][target_weather] = target_intensity
        _pgm, mask = generator(_pgm, _target_label, get_mask=True)

        _pgm = _pgm.detach()[0].numpy()
        mask = mask.detach()[0].numpy()
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 0, 1)
        _pgm = np.swapaxes(_pgm, 0, 2)
        _pgm = np.swapaxes(_pgm, 0, 1)

        pgm = pgm * (1 - mask[:, :, 0])
        return pgm,mask[:, :, 0]

    def get_masks_from_pgms(self, pgm, target_weather, target_intensity):
        _pgm = torch.tensor(pgm, dtype=torch.float32)
        _pgm = torch.zeros(16,1,32,1024)
        _target_label = torch.zeros((16,4))
        _pgm = torch.tensor(pgm)
        _target_label[:,3] = 0.3

        _pgm, mask = generator(_pgm, _target_label, get_mask=True)

        _pgm = _pgm.detach().numpy()
        mask = mask.detach().numpy()
        mask = np.swapaxes(mask, 1, 3)
        mask = np.swapaxes(mask, 1, 2)
        _pgm = np.swapaxes(_pgm, 1, 3)
        _pgm = np.swapaxes(_pgm, 1, 2)

        for i in range(16):
            print(f'{target_weather[i]} WITH {target_intensity[i]}')
            A = pgm[i] * (1 - mask[i, :, :, 0])
            plt.imshow(A.swapaxes(0,2).swapaxes(0,1))
            plt.axis('off')
            plt.show()

        return pgm,mask

    def mask_bin_and_label(self, mask, bin, label):
        num_points = max(bin.shape[0], bin.shape[1])
        result_bin = bin
        original_bin_with_random_intensity = bin
        result_label = label

        # PGM Generation phase
        for i in range(num_points - 1,0,-1):
            # Remove truck (my mistake, ego vehicle is Toyota's pc has label 0, not 18)
            if result_label[i] == 18:
                result_label[i] = 0

            x = bin[i][self.x_ind]    #
            y = bin[i][self.y_ind]    #   CAMBIA QUESTI INDICI per swappare gli assi
            z = bin[i][self.z_ind]    #   Per KITTI usare (0,1,2) come indici

            d = pow(x * x + y * y + z * z, 0.5)
            pitch = 0.0

            if d > 0.0001:
                pitch = math.asin(z / d)
            if d > self.far_plane:
                d = self.far_plane

            if pitch > self.v_fov_h or pitch < self.v_fov_l:
                continue

            yaw = self.get_centered_yaw(x, y)
            nu = (yaw - self.h_fov / 2.0) / (- self.h_fov)

            # Seleziona se la coordinata y del pixel è calcolata linearmente rispetto all'asse verticale del LiDAR
            # oppure se è non lineare secondo qualche distribuzione nota (KITTI è lineare)
            nv = self.get_nv_from_pitch(pitch, [])
            #nv = self.get_nv_from_list_distribution(pitch)

            if nu > 1.0 or nu < 0.0:
                continue
            if nv > 1.0 or nv < 0.0:
                continue

            u = nu * self.width
            v = nv * self.height
            result_bin[i][3] = 255.0 * (self.far_plane - d)/ self.far_plane
            result_bin[i][3] += random.gauss(0, 5.0)
            result_bin[i][3] = np.clip(result_bin[i][3], 0.0, 255.0)
            original_bin_with_random_intensity[i][3] = result_bin[i][3]

            if mask[int(v),int(u)] > 0.01:
                result_bin = np.delete(result_bin, i, axis=0)
                result_label = np.delete(result_label, i, axis=0)
        return result_bin, result_label, original_bin_with_random_intensity


if __name__ == '__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(device)
    sequence = '10'

    # Select generator
    # g_dimension, n_classes, residual_layers, batch, img_dimension
    generator = Generator_NoVerticalReduction(64, 4, 5, 16, (32, 1024))
    generator.eval()
    generator.load_state_dict(torch.load('models/m20/Ray_8x381_7200.pth', map_location=torch.device(device))['G'])
    weather_generator = WeatherPGMGeneraor(generator, 0, 1, 2)
    """ ----------------------- """
    """ ----------------------- """
    """ VISUALIZE 3D POINTCLOUD """
    """ ----------------------- """
    # pgm = load_image_as_array("./TEST/Dataset/PGM/synth_5333.png")
    # pgm, mask = weather_generator.get_mask_from_pgm(pgm/255.0, target_weather=0, target_intensity=0)
    # pgm,_ = polar_to_pointcloud(pgm*255.0, max_distance=150)
    # visualize_pointcloud(pgm)
    """ ----------------------- """
    """ ----------------------- """
    """ --------------------------------------------- """
    """ VISUALIZE PGM and LABELS from .bin and .label """
    """ --------------------------------------------- """
    #bin_file = np.fromfile("sequences/5000/velodyne/0000004.bin", dtype=np.float32)
    #bin_file = np.fromfile("Synth_19_10.bin", dtype=np.float32)
    #label_file = np.fromfile("sequences/5000/labels/0000004.label", dtype=np.uint32)
    #bin_file = bin_file.reshape((-1,4))
    #bin_file = bin_file[:,0:3]
    #n_points = len(bin_file)

    #pgm, _, original_zeros = weather_generator.bin_to_pgm(bin_file, label_file, as_numpy_array=True)
    #plt.imshow(pgm)
    #plt.show()
    #plt.imshow(_)
    #plt.show()

    #pgm, mask = weather_generator.get_mask_from_pgm(pgm,3,0.98)
    #plt.imshow(pgm * original_zeros)
    #plt.show()
    #plt.imshow(_/255.0 * (1 - mask[:, :, None]))
    #plt.show()

    """ ------------------------------------------- """
    """ ------------------------------------------- """
    """ ------------------------------------------- """
    """ ENHANCE Entire folder with selected weather """
    """ ------------------------------------------- """
    if os.path.exists(f'sequences/results/{sequence}/labels/'): shutil.rmtree(f'sequences/results/{sequence}/velodyne_with_random_intensity/')
    if os.path.exists(f'sequences/results/{sequence}/velodyne/'): shutil.rmtree(f'sequences/results/{sequence}/velodyne/')
    if os.path.exists(f'sequences/results/{sequence}/labels/'): shutil.rmtree(f'sequences/results/{sequence}/labels/')
    os.mkdir(f'sequences/results/{sequence}/velodyne_with_random_intensity/')
    os.mkdir(f'sequences/results/{sequence}/velodyne/')
    os.mkdir(f'sequences/results/{sequence}/labels/')

    i = 0
    n_file = len(os.listdir(f"sequences/{sequence}/velodyne/"))

    for file in os.listdir(f"sequences/{sequence}/velodyne/"):
        print(f'{i}/{n_file}')
        i += 1
        file_name = file.split('.')[0]
        bin_file = np.fromfile(f"sequences/{sequence}/velodyne/{file_name}.bin", dtype=np.float32)
        label_file = np.fromfile(f"sequences/{sequence}/labels/{file_name}.label", dtype=np.uint32)
        bin_file = bin_file.reshape((-1, 4))    # [n_punti, x,y,z,i]

        pgm, pgm_label, original_zeros = weather_generator.bin_to_pgm(bin_file, label_file, as_numpy_array=True)
        
        # RANDOM WEATHER IN OUTPUT
        random_label = random.randint(0,4)
        random_intensity = random.uniform(0.1,0.98)
        
        weathers = ['sun','rain','fog','snow']
        print(f'Is {weathers[random_label]} with intensity: {random_intensity}')
        pgm, mask = weather_generator.get_mask_from_pgm(pgm, target_weather=random_label, target_intensity=random_intensity)
        mask = mask * original_zeros
        new_bin, new_label, original_bin_with_random_intensity = weather_generator.mask_bin_and_label(mask, bin_file, label_file)

        new_bin.tofile(f'sequences/results/{sequence}/velodyne/{file_name}.bin')
        new_label.tofile(f'sequences/results/{sequence}/labels/{file_name}.label')
        original_bin_with_random_intensity.tofile(f'sequences/results/{sequence}/velodyne_with_random_intensity/{file_name}.bin')