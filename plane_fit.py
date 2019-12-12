# coding: utf-8
import slice as sl
import numpy as np
import pandas as pd
cs, mesh, __ = sl.slice_mesh('D-69057.stl')
points = pd.DataFrame(mesh.vertices, columns=['x','y','z'])
xy1 = np.concatenate([points.iloc[:,:-1], np.ones((points.shape[0],1))],axis=1)
z= np.array(points.iloc[:,-1]).reshape(-1,1)
fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(xy1.T, xy1)), xy1.T), z)
fit
