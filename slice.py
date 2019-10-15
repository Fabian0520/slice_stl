'''
 slice = mesh.section(plane_origin=mesh.centroid, plane_normal=[0,0,1])
 slice = mesh.section(plane_origin=([0,0,1]), plane_normal=[0,0,1])
 slice_2D, to_3D = slice.to_planar()
 slice_2D.show()

 if we wanted to take a bunch of parallel slices, like for a 3D printer
 we can do that easily with the section_multiplane method
 we're going to slice the mesh into evenly spaced chunks along z
 this takes the (2,3) bounding box and slices it into [minz, maxz]


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


 DataFrame maskieren:
   mask = (condition1) & (condition2) | slice_at_min_z_df['x'] > -40
 Maske anwenden:
   slice_at_min_z_df['x'][mask]

 ax = fig.add_subplot(1, 1, 1, aspect=1)

'''

import glob
import trimesh
import numpy as np
# from shapely.geometry import LineString
import pandas as pd
import matplotlib.pyplot as plt

mesh_file = '/home/fabian/Projekte/Python/vtk/D-68966.stl'


def slice_mesh(mesh_file, location=None, direction=[0, -1, 0]):
    mesh = trimesh.load_mesh(mesh_file)
    # Funktoniert mometan nur, wenn kein Pfad angegeben wird.
    mesh_name = mesh_file.split('.')[0]
    points = pd.DataFrame(mesh.vertices,
                         columns=pd.MultiIndex.from_product([[mesh_name],
                                                             ['x', 'y', 'z']]))
    # minimum der z-Spalte:
    if not location:
        # Location auf Minimum der Z-Koordinate
        # .idxmin liefert Zeile des Minimums in der angegebenen Spalte
        location = points.iloc[points[(mesh_name, 'z')].idxmin()]
    # z-Position = 0, damit der Rand auf 0 Niveau ist
    cross_section = mesh.section(plane_origin=[location[0],
                                            location[1],
                                            0], plane_normal=direction)

    # MultiColumns: MeshName und darunter Koordinaten
    columns = pd.MultiIndex.from_product([[mesh_name], ['x', 'y', 'z']])
    # sort_values(['x']), damit beim lineplot die Punkte richtig liegen.
    cross_section_df = pd.DataFrame(cross_section.vertices,
                                 columns=columns).sort_values(by=[(mesh_name, 'x')]).reset_index(drop=True)
    return cross_section_df, mesh, location

# sphere = trimesh.creation.icosphere(subdivisions=7)
# sphere.apply_translation((x,y,z))
# trimesh.nsphere.fit_nsphere(points)

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
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def fit_circle_3d(x, y, z, w=[]):

    A = np.array([x, y, z, np.ones(len(x))]).T
    b = x**2 + y**2 + z**2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    zc = c[2]/2
    r = np.sqrt(c[2] + xc**2 + yc**2 + zc**2)
    return xc, yc, zc, r

def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f, rcond=None)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return float(radius), float(C[0]), float(C[1]), float(C[2])

def plot_circle(x, y, radius=0.15):
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke
    circle = Circle((x, y), radius, clip_on=True, zorder=10, linewidth=1,
                    edgecolor='black', facecolor=(0, 0, 0, .0125),
                    path_effects=[withStroke(linewidth=5, foreground='w')],
                    alpha=0.1)
    # ax.add_artist(circle)
    return circle


def normalize_x(data):
    if type(data.columns) == 'pandas.core.indexes.bas.Index':   # nur eine Slice
        column_names = data.columns[0]
    else:
        column_names = data.columns.levels[0]
    for name in column_names:
        delta_loc = data[(name, 'z')].idxmin()
        delta_val = data[(name, 'x')][delta_loc]
        if delta_val > 0:
            data[(name, 'x')] = data[(name, 'x')] - abs(delta_val)
        else:
            data[(name, 'x')] = data[(name, 'x')] + abs(delta_val)
    return data


def plot_slices(data, aspect_ratio=1):
    fig, ax = plt.subplots(figsize=(11.6929, 8.26772))   # Din A4 Größe in inch Landscape
    ax.set_aspect(aspect=aspect_ratio)
    ax.set_title('Schnitte durch den tiefsten Punkt')
    ax.set_ylabel('Z [mm]')
    ax.set_xlabel('X [mm]')
    if type(data.columns) == 'pandas.core.indexes.bas.Index':   # nur eine Slice
        column_names = data.columns[0]
    else:
        column_names = data.columns.levels[0]
    for name in column_names:
        ax.scatter(data[(name, 'x')], data[(name, 'z')],
                   label=(name + '\n'r'min_z = '+'{:6.2f} mm'.format(data[(name, 'z')].min())), s=0.5)
    ax.legend(markerscale=6,
              scatterpoints=1,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.5),
              fancybox=True, ncol=3)
    ax.grid()
    fig.tight_layout()
    fig.show()
    fig.savefig('output.png', orientation='landscape', papertype='a4', dpi=600)
    return fig, ax

def fit_circle_xz(data):
    if type(data.columns) == 'pandas.core.indexes.bas.Index':   # nur eine Slice
        column_names = data.columns[0]
    else:
        column_names = data.columns.levels[0]
    for name in column_names:
        mask = data[(name, 'z')] < (0.0 + 0.5)  # 1 ist der Margin
        x, z, r = fit_circle_2d(data[(name, 'x')][mask], data[(name, 'z')][mask])
        abstand_daten_fit = np.abs( np.sqrt((data[(name, 'x')]-x)**2 + (data[(name, 'z')]-z)**2) - r)
        mask = abstand_daten_fit < 5
        x, z, r = fit_circle_2d(data[(name, 'x')][mask], data[(name, 'z')][mask])
    return x, z, r, abstand_daten_fit

def read_slices():
    data = pd.DataFrame()
    for file in glob.glob('*.json'):
        data = pd.concat([data, pd.read_json(file)], axis=1, sort=False)
    return data


# def write_slices(data):


if __name__ == '__main__':
    files = glob.glob('*.stl')
    print(files)
    cross_sections = pd.DataFrame()
    for file in files:
        cs, __, __ = slice_mesh(file)
        normalize_x(cs)
        cross_sections = pd.concat([cross_sections, cs], axis=1)
    plot_slices(cross_sections)


'''
1, Mesh einlesen.
2, Mesh maskieren (z-Koordinate < -0.5)
3, trimesh.nsphere.fit_nsphere(mesh[mask) fittet eine Kugel an die Punkte (gibt nur Koordinaten aus)
4, Abstand Kugel - Punkte bestimmen, neue Maske und neuer Fit. (evtl Abweichung von Fit-Fkt einbeziehen)
5, Kukel erzeugen: sphere = trimesh.creation.icosphere(subdivisions=7, radius=r_fit) (im Ursprung)
6, Kugel mit Mesh ausrichten: sphere.apply_translation(x,y,z aus fit)
7, Schnitt durch Mesh und Kugel (durch Mittelpunkt Kugel)
8, Schnitte in DataFrames umwandeln und plotten
'''
