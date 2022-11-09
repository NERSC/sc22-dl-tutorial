import ast
import matplotlib.pyplot as plt
import statistics
import json
import numpy as np

def return_avg_ips(file_loc):
    avg_ips = 0
    steps = 0
    myfile = open(file_loc, 'r')
    myline = myfile.readline().strip()
    while myline:
        stud_obj = json.loads(myline)
        io = stud_obj["IO"]["trial_throughput"][0]
        syth = stud_obj["SYNTHETIC"]["trial_throughput"][0]
        full = stud_obj["FULL"]["trial_throughput"][0]
        myline = myfile.readline().strip()
    
    myfile.close()
    return io, syth, full


n1 = return_avg_ips('weak_scale_1.json')
n2 = return_avg_ips('weak_scale_2.json')
n4 = return_avg_ips('weak_scale_4.json')
n8 = return_avg_ips('weak_scale_8.json')
n16 = return_avg_ips('weak_scale_16.json')
n32 = return_avg_ips('weak_scale_32.json')

# Real
io = [n1[0], n2[0], n4[0], n8[0], n16[0], n32[0]]
syth = [n1[1], n2[1], n4[1], n8[1], n16[1], n32[1]]
full = [n1[2], n2[2], n4[2], n8[2], n16[2], n32[2]]

# Ideal 
io_id = [n1[0], n1[0]*2, n1[0]*4, n1[0]*8, n1[0]*16, n1[0]*32]
syth_id = [n1[1], n1[1]*2, n1[1]*4, n1[1]*8, n1[1]*16, n1[1]*32]
full_id = [n1[2], n1[2]*2, n1[2]*4, n1[2]*8, n1[2]*16, n1[2]*32]

bX = [1, 2, 4, 8, 16, 32]

fig, axarr = plt.subplots(1,1)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        }

axarr.plot(bX, full, '-o', color='g', label='real data')
axarr.plot(bX, syth, '-o', color='darkorange', label='synthetic data')
axarr.plot(bX, io, '-o', color='darkblue', label='data only')
axarr.plot(bX, full_id, '--o', color='g', label='real data ideal')
axarr.plot(bX, syth_id, '--o', color='darkorange', label='synthetic data ideal')
axarr.plot(bX, io_id, '--o', color='darkblue', label='data only ideal')
axarr.set_ylabel('Avg. Throughput (img/sec)',fontdict=font)
axarr.set_xlabel('Nodes',fontdict=font)
axarr.set_title('3D U-Net Cosmo Weak Scaling', fontdict=font)
axarr.legend()
axarr.grid()

plt.show()
