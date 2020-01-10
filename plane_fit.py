# coding: utf-8
import slice as sl
import numpy as np
import pandas as pd
cs, mesh, __ = sl.slice_mesh('D-69023.stl')
points = pd.DataFrame(mesh.vertices, columns=['x','y','z'])
xy1 = np.concatenate([points.iloc[:,:-1], np.ones((points.shape[0],1))],axis=1)
z= np.array(points.iloc[:,-1]).reshape(-1,1)
fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(xy1.T, xy1)), xy1.T), z)
fit

'''
trimesh.points.plane_fit fittet plane an Punkte
trimesh.geometry.plane_transform(ursprung, vector) liefert Transformationsmatrix um Objekt nach Plane auszurichten
Objekt.apply_transform transformiert das Objekt

1)  trimesh.points.plane_fit(mesh.vertices)
2)  Transformationsmatrix berechnen
3)  Objekt drehen
4)  neuer plane_fit mit maskierten Punkten (evlt noch trimesh.points.points_plane_distance ansehen)
    -> 2 bis 4
5)  trimesh.nsphere.fit_nsphere()
