#!./venv/bin/python
import glob
import trimesh
import numpy as np
# from shapely.geometry import LineString
import pandas as pd
import matplotlib.pyplot as plt
import plane_fit

def slice_mesh(mesh, mesh_name, fit, location=None, direction=[0, -1, 0]):
    '''
        Schneidet das Netz mit einer Ebene (Ort und Richtung
        können definiert werden).
        mesh: zu schneidendes Netz [trimesh.mesh]
        mesh_name: Name des Netzes. Wird für den Index der Slices verwendet [str]
        fit: Parameter der gefitteten Kugel (aus plane_fit.py). Werden auch in ausgabe gespeichert.
        location: Punkt der Schnittebene [array(3)]
        direction: Normale der Schnittebene (default: x-z Ebene) [array(3)]
    '''

    points = pd.DataFrame(mesh.vertices,
                         columns=pd.MultiIndex.from_product([[mesh_name],
                                                             ['x', 'y', 'z']]))
    if not location:
        # Location auf Minimum der Z-Koordinate
        # .idxmin liefert Zeile des Minimums in der angegebenen Spalte
        location = points.iloc[points[(mesh_name, 'z')].idxmin()]
    cross_section = mesh.section(plane_origin=[location[0],
                                            location[1],
                                            0], plane_normal=direction)

    # MultiColumns: MeshName und darunter Koordinaten
    columns = pd.MultiIndex.from_product([[mesh_name], ['x', 'y', 'z']])
    # sort_values(['x']), damit beim lineplot die Punkte richtig liegen.
    cross_section_df = pd.DataFrame(cross_section.vertices,
                                 columns=columns).sort_values(by=[(mesh_name, 'x')]).reset_index(drop=True)
    r_crater = np.sqrt(fit[1]**2 - fit[0][2]**2)    # calculate crater radius
    cross_section_df[(mesh_name, 'min_z')] = pd.Series(points.iloc[points[(mesh_name, 'z')].idxmin()].values)
    cross_section_df[(mesh_name, 'max_z')] = pd.Series(points.iloc[points[(mesh_name, 'z')].idxmax()].values)
    cross_section_df[(mesh_name, 'sph_center')] = pd.Series(fit[0])
    cross_section_df[(mesh_name, 'sph_radius')] = pd.Series(fit[1])
    cross_section_df[(mesh_name, 'sph_error')] = pd.Series(fit[2])
    cross_section_df[(mesh_name, 'r_crater')] = pd.Series(r_crater)
    return cross_section_df, location

def plot_slices(data, aspect_ratio=1):
    '''
        Plottet die Schnitte durch die Netze. fig2 sind die Einzelplots, fig sind
        die Übersihtsplots (falls mehrer Slices übergeben werden).

        data: zu plottende Date. [pd.DataFrame]
        aspect_ratio: Aspect ratio der Axen
    '''

    fig, ax = plt.subplots(figsize=(11.6929, 8.26772))   # Din A4 Größe in inch Landscape
    ax.set_aspect(aspect=aspect_ratio)
    ax.set_title('Schnitte durch Zentrum Krater')
    ax.set_ylabel('Z [mm]')
    ax.set_ylim(-35,15)
    ax.set_xlabel('X [mm]')
    if type(data.columns) == 'pandas.core.indexes.bas.Index':   # nur eine Slice
        column_names = data.columns[0]
    else:
        column_names = data.columns.levels[0]
    for name in column_names:
        #   Einzelplotts
        #----------------------------------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(11.6929, 8.26772)) #480/my_dpi, ..., dpi=my_dpi
        ax2.set_aspect(aspect=aspect_ratio)
        ax2.set_title('Schnitte durch Zentrum Krater')
        ax2.set_ylabel('Z [mm]')
        ax2.set_ylim(-35,15)
        ax2.set_xlabel('X [mm]')
        ax2.scatter(data.xs('x',level=1,axis=1), data.xs('z',level=1,axis=1), color='gray', s=0.1, alpha=0.4) # select in a multiindex df in the second level
        min_z_sph = data[(name, 'sph_center')][2] - data[(name, 'sph_radius')][0]
        min_z = data[(name, 'min_z')][2]
        label = (name + '\n' +
                 'min_z = {0:6.2f} mm \nmin_z_sph = {1:6.2f} mm'.format(min_z, min_z_sph))
        ax2.scatter(data[(name, 'x')], data[(name, 'z')],
                   label=label, s=0.5)
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
        min_z_sph = data[(name, 'sph_center')][2] - data[(name, 'sph_radius')][0]
        min_z = data[(name, 'min_z')][2]
        label = (name + '\n' +
                 'min_z = {0:6.2f} mm \nmin_z_sph = {1:6.2f} mm'.format(min_z, min_z_sph))
        ax.scatter(data[(name, 'x')], data[(name, 'z')],
                   label=label, s=0.5)
        # ax.add_artist(plot_circle(0,data[(name,'sph_center')][2],data[(name,'sph_radius')][0]))
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
    '''
        Erzeugt die Kreise für plot_slices().
    '''

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

#data = pd.read_csv('file', header=[0,1])

if __name__ == '__main__':
    files = glob.glob('*.stl')
    print(files)
    cross_sections = pd.DataFrame()
    for file in files:
        mesh_name = file.split('.')[0]
        output_name = mesh_name + '.csv'
        mesh, sphere_fit_parameters = plane_fit.prepare_and_fit(file)
        cs, __ = slice_mesh(mesh, mesh_name, sphere_fit_parameters, location=[0,0])
        cs.to_csv(output_name, index=False)
        # führt alle Schnitte in einem Dataframe zusammen, damit dann alle geplottet werden können
        cross_sections = pd.concat([cross_sections, cs], axis=1)
    plot_slices(cross_sections)
