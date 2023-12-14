from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import optparse
import numpy as np
import matplotlib.pyplot as plt

from decimal import *

Peak = 19.5 # TFLOPS
TB = 1024 * 1024 * 1024 * 1024

def draw_line_chart(methods, dims, data, figure_name, y_label, title):
  fig = plt.figure(figsize=(32, 24), dpi=100)

  dims_str = list(map(str, dims))

  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
  linestyles = ['-', '--', '-.', ':']

  for i in range(len(methods)):
    plt.plot(dims_str, data[i], color=colors[i % len(colors)],
      linestyle=linestyles[(i // len(colors)) % len(linestyles)], marker='o', markersize=6)

    # plt.xticks(dims)
    plt.ylim(bottom=0)
    plt.yticks(range(0, round(np.max(np.max(data, axis=0)) + 0.5) + 10, 10))
    plt.tick_params(labelsize=25)

    # plt.hlines(y=100, xmin=dims_str[0], xmax=dims_str[-1], colors='r', linestyles='-.')
    plt.grid(True, linestyle='-.')

    plt.xlabel('Matrix Dimension / M = N = K', fontdict={'size': '30'})
    plt.ylabel(y_label, fontdict={'size': '30'})
    plt.title(title, fontdict={'size': '30'})
    plt.legend(methods, loc='best', prop={'size': '30'})

    plt.savefig(figure_name, dpi=fig.dpi)
    # plt.show()

def read_data(methods, dims, file):
  global Peak
  global TB
  data_latency = np.zeros((len(methods), len(dims)), np.float64)
  data_throughtput = np.zeros((len(methods), len(dims)), np.float64)
  peak_ratio = np.zeros((len(methods), len(dims)), np.float64)
  
  with open(file) as fp:
    index = -2
    next_line = fp.readline()
    while next_line:
      line = next_line
      next_line = fp.readline()
      if "Matrix size" in line:
        index += 1
      elif 'My' in line and index >= 0:
        iterms = line.split(' ')
        lantency = Decimal(iterms[4])
        dim = dims[index]
        data_latency[0][index] = lantency
        data_throughtput[0][index] = Decimal(2 * dim * dim * dim) / Decimal(TB) / (lantency / Decimal(1000))  # 2MNK
        peak_ratio[0][index] = data_throughtput[0][index] / Peak * 100
      elif "cublas Latency(NN)" in line and index >= 0:
        iterms = line.split(' ')
        lantency = Decimal(iterms[3])
        dim = dims[index]
        data_latency[1][index] = lantency
        data_throughtput[1][index] = Decimal(2 * dim * dim * dim) / Decimal(TB) / (lantency / 1000)  # 2MNK
        peak_ratio[1][index] = data_throughtput[1][index] / Peak * 100
      elif "cublas Latency(TT)" in line and index >= 0:
        iterms = line.split(' ')
        lantency = Decimal(iterms[3])
        dim = dims[index]
        data_latency[2][index] = lantency
        data_throughtput[2][index] = Decimal(2 * dim * dim * dim) / Decimal(TB) / (lantency / 1000)  # 2MNK
        peak_ratio[2][index] = data_throughtput[2][index] / Peak * 100
        
        lantency = Decimal(2 * dim * dim * dim) / Decimal(TB) / Decimal(Peak * 1000) # ms
        data_latency[3][index] = lantency
        data_throughtput[3][index] = Peak
        peak_ratio[3][index] = 1 * 100
      
  return data_latency, data_throughtput, peak_ratio

def main(file):
  methods = ["Generated-Matmul", "Cublas-NN", "Cublas-TT", "A100-PCIe-Peak"]
  dims = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
  data_latency, data_throughtput, peak_ratio = read_data(methods, dims, file)
  draw_line_chart(methods, dims, data_latency, 'latency.png', 'Latency / ms', 'Matmul Lantency')
  draw_line_chart(methods, dims, data_throughtput, 'throughput.png', 'Throughput / TFLOPS', 'Matmul Throughput')
  draw_line_chart(methods, dims, peak_ratio, 'peak_ratio.png', 'Performance Compared with Peak TPUT / %', 'Matmul Performance')

  

if __name__ == "__main__":
  main("./output.log")

