#!./venv/bin/python
import glob
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plane_fit
import pickle
from dc_object import DataCraterAnalysis

def slice_mesh(mesh, location=[0,0,0], direction=[0,1,0]):
    '''
        Schneidet das Netz mit einer Ebene (Ort und Richtung
        können definiert werden).
        mesh: zu schneidendes Netz [trimesh.mesh]
        mesh_name: Name des Netzes. Wird für den Index der Slices verwendet [str]
        fit: Parameter der gefitteten Kugel (aus plane_fit.py). Werden auch in ausgabe gespeichert.
        location: Punkt der Schnittebene [array(3)]
        direction: Normale der Schnittebene (default: x-z Ebene) [array(3)]
    '''

    cross_section = mesh.section(plane_origin=location, plane_normal=direction)
    columns = ['x', 'y', 'z']
    # sort_values(['x']), damit beim lineplot die Punkte richtig liegen.
    cross_section_df = pd.DataFrame(cross_section.vertices, columns=columns).sort_values(by=['x']).reset_index(drop=True)
    loc_dir = pd.DataFrame([location,direction], index=['location','direction'], columns=['x','y','z'])
    cross_section_df = pd.concat([loc_dir,cross_section_df])

    return cross_section_df

def plot_slices(data, aspect_ratio=1):
    '''
        Plottet die Schnitte durch die Netze. fig2 sind die Einzelplots, fig sind
        die Übersihtsplots (falls mehrer Slices übergeben werden).

        data: zu plottende Date. [pd.DataFrame]
        aspect_ratio: Aspect ratio der Axen
    '''

    #fig, ax = plt.subplots(figsize=(11.6929, 8.26772))   # Din A4 Größe in inch Landscape
    #fig, ax = plt.subplots()
    #ax.set_aspect(aspect=aspect_ratio)
    #todo Schnittachse
    #ax.set_title('Schnitte durch Zentrum Krater')
    #ax.set_ylabel('Z [mm]')
    #ax.set_ylim(-35,15)
    #ax.set_xlabel('X [mm]')
    for analysis in data:
        for n_slice in range(0,len(analysis.cross_section)):
            #   Einzelplotts
            #----------------------------------------------------------------------------------------
            fig2, ax2 = plt.subplots(figsize=(11.6929, 8.26772)) #480/my_dpi, ..., dpi=my_dpi
            ax2.set_aspect(aspect=aspect_ratio)
            direction = np.array(analysis.cross_section[n_slice].loc['direction'])
            location = np.array(analysis.cross_section[n_slice].loc['location'])
            #todo Schnittachse
            ax2.set_title(f"{analysis.name}\nSchnitte durch Punkt: {location}, Ebenennormale: {direction}")
            ax2.set_ylabel('z [mm]')
            ax2.set_ylim(-35,15)
            min_z_sph = float(analysis.fit['r'] - analysis.fit['z'])
            min_z = min(analysis.points['z'][2:])
            label = (f'{analysis.name} \nmin_z = {min_z:6.2f} mm \nmin_z_sph = {min_z_sph:6.2f} mm')
            if np.round(analysis.cross_section[n_slice]['x'][3]) == 0:
                ax2.set_xlabel('y [mm]')
                ax2.scatter(analysis.cross_section[n_slice]['y'][2:], analysis.cross_section[n_slice]['z'][2:], s=0.1, label=label) # [2:] weil in ersten beiden yeilen loc und dir stehen!
            elif np.round(analysis.cross_section[n_slice]['y'][3]) == 0:
                ax2.set_xlabel('x [mm]')
                ax2.scatter(analysis.cross_section[n_slice]['x'][2:], analysis.cross_section[n_slice]['z'][2:], s=0.1, label=label) # [2:] weil in ersten beiden yeilen loc und dir stehen!
            ax2.add_artist(plot_circle(0, analysis.fit['z'], analysis.fit['r']))
            ax2.legend(markerscale=6,
                      scatterpoints=1,
                      loc='upper center',
                      bbox_to_anchor=(0.5, -0.5),
                      fancybox=True, ncol=3)
            ax2.grid(linewidth=0.2, alpha=0.7, color='black')
            fig2.tight_layout()
            output_name = f'{analysis.name}_{n_slice}.png'
            fig2.savefig(output_name, orientation='landscape', papertype='a4', dpi=600)
            plt.close(fig2)
            #----------------------------------------------------------------------------------------
'''
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
'''
    #return fig, ax


def plot_contour(data):
    levels = 50
    fig = plt.figure()
    ax = fig.add_subplot(aspect=1)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f"{data.name}")
    ax.grid(linewidth=0.2, alpha=0.7, color='black')
    vmin = data.points['z'].min()
    vmax = data.points['z'].max()
    plot_val = np.linspace(vmin, vmax, levels, endpoint=True)
    cntr = ax.tricontourf(data.points['x'][::500],data.points['y'][::500],data.points['z'][::500], plot_val, vmin=vmin, vmax=vmax, extend='both')#, cmap='')
    ax.tricontour(data.points['x'][::50],data.points['y'][::50],data.points['z'][::50], levels,  linewidths=0.2, alpha=0.7, colors='black')
    cbar = fig.colorbar(cntr, ax=ax, label='z [mm]')
    output_name = f'{data.name}_contour.png'
    fig.savefig(output_name, orientation='landscape', papertype='a4', dpi=600)
    plt.close(fig)

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

def read_pkl():
    from dc_object import DataCraterAnalysis
    files = glob.glob('*.pkl')
    c_a_list = []
    for f in files:
        c_a_list.append(pickle.load(open(f, 'rb')))
    return c_a_list

if __name__ == '__main__':
    files = glob.glob('*.stl')
    print(files)
    c_a_list = []
    for file in files:
        crater_analysis = DataCraterAnalysis()
        mesh_name = file.split('.')[0]
        print(f'processing {mesh_name}')
        output_name = mesh_name + '.csv'
        mesh, mesh_points, fit_parameters = plane_fit.prepare_and_fit(file)
        crater_analysis.name = mesh_name
        crater_analysis.points = mesh_points
        crater_analysis.fit = fit_parameters
        for direction in [[0,1,0],[1,0,0]]:
            cs = slice_mesh(mesh, direction=direction)
            crater_analysis.cross_section.append(cs)
        pickle.dump( crater_analysis, open(mesh_name+'.pkl','wb'))
        c_a_list.append(crater_analysis)
        plot_contour(crater_analysis)
    plot_slices(c_a_list)

#data = pd.read_csv('file', header=[0,1])
