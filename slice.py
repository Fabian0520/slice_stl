#!./venv/bin/python
import glob
import trimesh
import numpy as np
# from shapely.geometry import LineString
import pandas as pd
import matplotlib.pyplot as plt
import plane_fit

mesh_file = '/home/fabian/Projekte/Python/vtk/D-68966.stl'

def slice_mesh(mesh_file, mesh_name, location=None, direction=[0, -1, 0]):
    #mesh = trimesh.load_mesh(mesh_file)
    # Funktoniert mometan nur, wenn kein Pfad angegeben wird.
    #mesh_name = mesh_file.split('.')[0]
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
    return cross_section_df, location

def plot_slices(data, aspect_ratio=1):
    fig, ax = plt.subplots(figsize=(11.6929, 8.26772))   # Din A4 Größe in inch Landscape
    ax.set_aspect(aspect=aspect_ratio)
    ax.set_title('Schnitte durch Zentrum Krater')
    ax.set_ylabel('Z [mm]')
    ax.set_xlabel('X [mm]')
    if type(data.columns) == 'pandas.core.indexes.bas.Index':   # nur eine Slice
        column_names = data.columns[0]
    else:
        column_names = data.columns.levels[0]
    for name in column_names:
        #   Einzelplotts
        #----------------------------------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(11.6929, 8.26772))
        ax2.set_aspect(aspect=aspect_ratio)
        ax2.set_title('Schnitte durch Zentrum Krater')
        ax2.set_ylabel('Z [mm]')
        ax2.set_xlabel('X [mm]')
        ax2.scatter(data[(name, 'x')], data[(name, 'z')],
                   label=(name + '\n'r'min_z = '+'{:6.2f} mm'.format(data[(name, 'z')].min())), s=0.5)
        ax2.add_artist(plot_circle(0,data[(name,'sph_center')][2],data[(name,'sph_radius')][0]))
        ax2.legend(markerscale=6,
                  scatterpoints=1,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.5),
                  fancybox=True, ncol=3)
        ax2.grid()
        fig2.tight_layout()
        output_name = name + '.png'
        fig2.savefig(output_name, orientation='landscape', papertype='a4', dpi=600)
        plt.close(fig2)
        #----------------------------------------------------------------------------------------
        ax.scatter(data[(name, 'x')], data[(name, 'z')],
                   label=(name + '\n'r'min_z = '+'{:6.2f} mm'.format(data[(name, 'z')].min())), s=0.5)
        ax.add_artist(plot_circle(0,data[(name,'sph_center')][2],data[(name,'sph_radius')][0]))
    ax.legend(markerscale=6,
              scatterpoints=1,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.5),
              fancybox=True, ncol=3)
    ax.grid()
    fig.tight_layout()
    fig.savefig('output.png', orientation='landscape', papertype='a4', dpi=600)
    return fig, ax

def plot_circle(x,y, radius=0.15):
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke
    circle = Circle((x,y), radius, clip_on=True, zorder=10, linewidth=1,
                    edgecolor='black', facecolor=(0,0,0, .0125),
                    path_effects=[withStroke(linewidth=5, foreground='w')],
                    alpha=0.1)
    # ax.add_artist(circle)
    return circle

def read_slices():
    data = pd.DataFrame()
    for file in glob.glob('*.json'):
        data = pd.concat([data, pd.read_json(file)], axis=1, sort=False)
    return data

if __name__ == '__main__':
    files = glob.glob('*.stl')
    print(files)
    cross_sections = pd.DataFrame()
    for file in files:
        mesh_name = file.split('.')[0]
        output_name = mesh_name + '.csv'
        mesh, sphere_fit_parameters = plane_fit.prepare_and_fit(file)
        r_crater = np.sqrt(sphere_fit_parameters[1]**2 - sphere_fit_parameters[0][2]**2)
        print('Name: {0}, tiefster Punkt (Kugel): {1:4.2f}mm, Durchmesser Krater: {2:4.2f}mm'.format(mesh_name, sphere_fit_parameters[0][2]-sphere_fit_parameters[1],r_crater*2))
        cs, __ = slice_mesh(mesh, mesh_name, location=[0,0])
        #import ipdb; ipdb.set_trace()
        cs[mesh_name, 'sph_center'] = pd.Series(sphere_fit_parameters[0])   # sollte ja eigentlich auf 0,0 liegen ...
        cs[mesh_name, 'sph_radius'] = pd.Series(sphere_fit_parameters[1])
        cs[mesh_name, 'sph_error'] = pd.Series(sphere_fit_parameters[2])
        cs.to_csv(output_name, index=False)
        # führt alle Schnitte in einem Dataframe zusammen, damit dann alle geplottet werden können
        cross_sections = pd.concat([cross_sections, cs], axis=1)
        # cross_sections.to_csv('tmp.csv', index=False)
    plot_slices(cross_sections)
