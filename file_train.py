import torch
import os
from read_pmf import readPFM
import numpy as np


def file_train(file_path):
    file_path_list = []
    files = os.listdir(file_path)
    #print(files)
    #print(len(files))
    for i in files:
        file_path = "E:/monkaa__optical_flow/monkaa__optical_flow\optical_flow/eating_camera2_x2/into_future/left/"+i
        file_path_list.append(file_path)
        #a, b = readPFM(file_path)
        #a = np.ascontiguousarray(a)
        #a = torch.from_numpy(a)
    return file_path_list
    print(file_path_list)



def print_tensor(data, scale=1):
    data = np.ascontiguousarray(data)
    data = torch.from_numpy(data)
    print(data)
    print("------------------------------------------------")
    print(scale)



file_path = 'E:/monkaa__optical_flow/monkaa__optical_flow\optical_flow/eating_camera2_x2/into_future/left'
file_train(file_path)


