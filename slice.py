'''
# slice = mesh.section(plane_origin=mesh.centroid, plane_normal=[0,0,1])
# slice = mesh.section(plane_origin=([0,0,1]), plane_normal=[0,0,1])
# slice_2D, to_3D = slice.to_planar()
# slice_2D.show()

# if we wanted to take a bunch of parallel slices, like for a 3D printer
# we can do that easily with the section_multiplane method
# we're going to slice the mesh into evenly spaced chunks along z
# this takes the (2,3) bounding box and slices it into [minz, maxz]


def tmp():  # slice multiplane
    z_extents = mesh.bounds[:,0]
    # slice every .125 model units (eg, inches)
    z_levels  = np.arange(*z_extents, step=5.000)

    # find a bunch of parallel cross sections
    sections = mesh.section_multiplane(plane_origin=mesh.bounds[1],
                                       plane_normal=[0,-1,0],
                                       heights=z_levels)
    sections

    sections_clear = [x for x in sections if type(x) == trimesh.path.path.Path2D]
    # pd.DataFrame(sections_clear[0].vertices)

    combined = np.sum(sections_clear)

    combined.show()


# DataFrame maskieren:
#   mask = (condition1) & (condition2) | slice_at_min_z_df['x'] > -40
# Maske anwenden:
#   slice_at_min_z_df['x'][mask]

# ax = fig.add_subplot(1, 1, 1, aspect=1)

'''

import trimesh
import numpy as np
from shapely.geometry import LineString
import pandas as pd
import matplotlib.pyplot as plt
import argparse

mesh_file = '/home/fabian/Projekte/Python/vtk/D-68966.stl'


def slice_at_min_z(mesh_file):
    mesh = trimesh.load_mesh(mesh_file)
    poits = pd.DataFrame(mesh.vertices, columns=['x', 'y', 'z'])
    # minimum der z-Spalte:
    location_min_z = poits.iloc[poits['z'].idxmin()] # .idxmin liefert Zeile des Minimums in der angegebenen Spalte

    slice_at_min_z = mesh.section(plane_origin=[location_min_z[0],
                                                  location_min_z[1],
                                                  0], # z-Position = 0, damit der Rand auf 0 Niveau ist
                                                  plane_normal=[0, -1, 0])
    slice_at_min_z_df = pd.DataFrame(slice_at_min_z.vertices,
                                     columns=['x', 'y', 'z']).sort_values(by=['x'])
    # sort_values(['x']), damit beim lineplot die Punkte richtig liegen.
    return slice_at_min_z_df
'''
    um als json zu speichern:
        slice_at_min_z_df.do_json(orient='split')
    zum laden:
        df = pd.read_json(path, orient='split')
'''


def fit_circle_2d(x, y, w=[]):

    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W,A)
        b = np.dot(W,b)

             # Solve by method of least squares
    c = np.linalg.lstsq(A,b,rcond=None)[0]

             # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r


def circle(x, y, radius=0.15):
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke
    circle = Circle((x, y), radius, clip_on=True, zorder=10, linewidth=1,
                    edgecolor='black', facecolor=(0, 0, 0, .0125),
                    path_effects=[withStroke(linewidth=5, foreground='w')],
                    alpha=0.1)
    ax.add_artist(circle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--f",
                        nargs=1,
                        type=str,
                        dest='mesh')

    args = parser.parse_args()
    print(args.mesh)

    output = args.mesh[0].split('.')[0] + '.json'

    slice_ = slice_at_min_z(args.mesh[0])
    x,y,r = fit_circle_2d(slice_['x'], slice_['z'])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    circle(x,y,r)
    ax.plot(slice_['x'],slice_['z'])
    plt.show()
    slice_.to_json(output)
