import trimesh
import numpy as np
from shapely.geometry import LineString
import pandas as pd
import matplotlib.pyplot as plt
import argparse
#%pylab inline
#%matplotlib inline

mesh_file = '/home/fabian/Projekte/Python/vtk/D-68966.stl'


def slice_at_min_z(mesh_file):
    dir(trimesh)
    mesh = trimesh.load_mesh(mesh_file)
    poits = pd.DataFrame(mesh.vertices, columns=['x', 'y', 'z'])
    # minimum der z-Spalte:
    location_min_z = poits.iloc[poits['z'].idxmin()] # .idxmin liefert Zeile des Minimums in der angegebenen Spalte

    slice_at_min_z = mesh.section(plane_origin=[location_min_z[0],
                                                  location_min_z[1],
                                                  0], plane_normal=[0, -1, 0])
    slice_at_min_z_df = pd.DataFrame(slice_at_min_z.vertices, columns=['x', 'y', 'z']).sort_values(by=['x'])
    # sort_values(['x']), damit beim lineplot die Punkte richtig liegen.

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(slice_at_min_z_df['x'], slice_at_min_z_df['z'])
    plt.show()

# slice = mesh.section(plane_origin=mesh.centroid, plane_normal=[0,0,1])


# slice = mesh.section(plane_origin=([0,0,1]), plane_normal=[0,0,1])
# slice_2D, to_3D = slice.to_planar()
# slice_2D.show()

# if we wanted to take a bunch of parallel slices, like for a 3D printer
# we can do that easily with the section_multiplane method
# we're going to slice the mesh into evenly spaced chunks along z
# this takes the (2,3) bounding box and slices it into [minz, maxz]


def tmp():
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--f",
                        nargs=1,
                        type=str)

    args = parser.parse_args()
    print(args.f)

    slice_at_min_z(args.f[0])
