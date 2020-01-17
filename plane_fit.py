import trimesh
import numpy as np
import pandas as pd

#mesh = trimesh.load_mesh('D-69028.stl')
mesh = trimesh.load_mesh('D-69049.stl')
mesh_org = mesh.copy()

plane_fit = trimesh.points.plane_fit(mesh.vertices)
trans_matrix = trimesh.geometry.plane_transform(plane_fit[0], plane_fit[1])
mesh.apply_transform(trans_matrix)

mesh_points = pd.DataFrame(mesh.vertices, columns=['x', 'y', 'z'])  # convert vertices to pd.DF
# calculate histogram
hist_values, hist_bins = np.histogram(mesh_points['z'], bins=500, density=True)
# Location of maximum value in hist_values
hist_max_pos = hist_values.argmax()
import ipdb; ipdb.set_trace()
# mask points so they are near hist_max
mask = mesh_points['z'].between(hist_bins[hist_max_pos], hist_bins[hist_max_pos+1])
# new fit and transform
plane_fit = trimesh.points.plane_fit(mesh.vertices[mask])
trans_matrix = trimesh.geometry.plane_transform(plane_fit[0], plane_fit[1])
mesh.apply_transform(trans_matrix)

# fit sphere to points not in plane
mesh_points = pd.DataFrame(mesh.vertices, columns=['x', 'y', 'z'])  # generate new points, after transform
mask = mesh_points['z'] < -0.5 # take only points with z<0.5 *** wert noch automatisch eryeugen
sph_fit_c, sph_fit_r, sph_fit_err = trimesh.nsphere.fit_nsphere(mesh_points[mask])


import ipdb; ipdb.set_trace()
# Maske "Abstand Punkte von Kugel kleiner x" erzeugen und noch mal fitten
# Distance between points and sphere surface
mesh_points['dist_sph'] = np.absolute(np.linalg.norm(mesh_points.loc[:,'x':'z']-sph_fit_c, axis=1.0)-sph_fit_r)
mask = mesh_points['dist_sph'] < 3.0
sph_fit_c, sph_fit_r, sph_fit_err = trimesh.nsphere.fit_nsphere(mesh_points.loc[:,'x':'z'][mask])

import ipdb; ipdb.set_trace()

# create and aligning sphere and mesh
sphere = trimesh.creation.icosphere(subdivisions=4, radius=sph_fit_r)   # generate sphere at origin
trans_matrix_sphere = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,sph_fit_c[2]],[0,0,0,1]])    # align sphere and mesh in z axis
sphere.apply_transform(trans_matrix_sphere)
trans_matrix_mesh = np.array([[1,0,0,-sph_fit_c[0]],[0,1,0,-sph_fit_c[1]],[0,0,1,0],[0,0,0,1]]) # align mesh and sphere in xy plane, so that origin is colinear with c_sphere
mesh.apply_transform(trans_matrix_mesh)

mesh_points = pd.DataFrame(mesh.vertices, columns=['x', 'y', 'z'])  # generate new points, after transform
mesh_points['dist_sph'] = np.absolute(np.linalg.norm(mesh_points.loc[:,'x':'z']-sph_fit_c, axis=1.0)-sph_fit_r)

#test.export('fit.stl')
