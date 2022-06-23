from plyfile import PlyData
import numpy as np
import networkx as nx
from scipy import sparse
from plyfile import PlyElement

from scattering_utility import *

def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = np.transpose([np.array(data['vertex'][axis]) for axis in ['x', 'y', 'z']])

    faces = data['face']['vertex_indices']
    faces = [np.array(face) for face in faces]
    face = np.transpose(faces)

    return pos,face

def read_shot(file):
    with open(file,'r') as f:
        data = f.readlines()
    feature = np.zeros((6890,352))
    for i in range(len(data)-1):
        line = data[i]
        vertex_index = int(line.split()[2][4:])
        feature[vertex_index] = np.array(line.split()[6:],dtype=np.float32)
    return feature

K = 80
norm = [1, 2, 3, 4]
J = 8
all_features = []
# choose parameter epsilon to be the 75th percentile of pairwise distances
this_eps = "auto"
q = 0.75

for t in range(100):
    # missing SHOT 90
    if t == 90:
        continue
    if t<10:
        pos,face= read_ply('FAUST_FILE/tr_reg_'+'00'+str(t)+'.ply')
    else:
        pos,face= read_ply('FAUST_FILE/tr_reg_'+'0'+str(t)+'.ply')
    N = pos.shape[0]
    vals, vecs, eps = compute_eigen(pos, this_eps, K, d=2, eps_quantile=q)
    shot = read_shot('SHOT/'+'shot'+str(t)+'.txt')
    F = compute_all_features(vals, vecs, shot, eps, N, norm, J)
    np.savetxt('FAUST_FEATURES/' + 'feature' + str(t) + '.txt',F)
    print(t)
    
