#!./venv/bin/python
import pathlib
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plane_fit
import pickle
from dc_object import DataCraterAnalysis
import generate_report

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
    # erzeugt einen String aus den Listen für direction und location
    dir_string = ''.join(map(str, direction))
    loc_string = ''.join(map(str, location))
    columns = pd.MultiIndex.from_product([[dir_string],['x', 'y', 'z']])
    # sort_values(['x']), damit beim lineplot die Punkte richtig liegen.
    cross_section_df = pd.DataFrame(cross_section.vertices, columns=columns).reset_index(drop=True).sort_values(by=(dir_string,'x'))

    return cross_section_df

def plot_slices(data, aspect_ratio=1):
    '''
        Plottet die Schnitte durch die Netze. fig2 sind die Einzelplots, fig_all sind
        die Übersihtsplots (falls mehrer Slices übergeben werden).

        data: zu plottende Date. [pd.DataFrame]
        aspect_ratio: Aspect ratio der Axen
    '''
    # minima und maxima von allen punkten finden (also von allen netzen)
    # eigene Funktion schreiben?
    min_all = pd.concat( [ data[i].points for i in range(0, len(data)) ] ).min()
    max_all = pd.concat( [ data[i].points for i in range(0, len(data)) ] ).max()
    d_all = abs(min_all) + max_all
    plot_range_all = pd.DataFrame([], index=['x','y','z'])
    plot_range_all = pd.concat([min_all, max_all], axis=1)    #z achse sollte groesseren abstand haben!
    plot_range_all.columns = ['min','max']
    plot_range_all['min']['x':'y'] = plot_range_all['min']['x':'y'] - d_all['x':'y']*0.05
    plot_range_all['max']['x':'y'] = plot_range_all['max']['x':'y'] + d_all['x':'y']*0.05
    plot_range_all['min']['z'] = plot_range_all['min']['z'] - d_all['z']*0.2
    plot_range_all['max']['z'] = plot_range_all['max']['z'] + d_all['z']*0.2

    fig_all, ax_all = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 9))   # Din A4 Größe in inch Landscape
    ax_all[0].set_aspect(aspect=aspect_ratio)
    ax_all[1].set_aspect(aspect=aspect_ratio)
    ax_all[0].set_ylabel('Z [mm]')
    ax_all[1].set_ylabel('Z [mm]')
    ax_all[0].set_xlim([ plot_range_all[col]['x'] for col in plot_range_all.columns ])
    ax_all[1].set_xlim([ plot_range_all[col]['y'] for col in plot_range_all.columns ])
    ax_all[0].set_ylim([ plot_range_all[col]['z'] for col in plot_range_all.columns ])
    ax_all[1].set_ylim([ plot_range_all[col]['z'] for col in plot_range_all.columns ])
    ax_all[0].grid(linewidth=0.2, alpha=0.7, color='black')
    ax_all[1].grid(linewidth=0.2, alpha=0.7, color='black')

    for analysis in data:
        z_min = min(analysis.points['z'])
        min_z_sph = float((analysis.fit['r'] - analysis.fit['z']) * (-1))
        fig2, ax_single = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 9))
        #fig2.suptitle(f"{analysis.name}")

        ax_single[0].set_aspect(aspect=aspect_ratio)
        ax_single[0].set_ylim([ plot_range_all[col]['z'] for col in plot_range_all.columns ])
        ax_single[0].set_ylabel('Z [mm]')
        ax_single[0].set_title(f"Schnitte durch die X-Z Ebene")
        ax_single[0].set_xlim([ plot_range_all[col]['x'] for col in plot_range_all.columns ])
        ax_single[0].set_xlabel('X [mm]')
        ax_single[0].add_artist(plot_circle(0, analysis.fit['z'], analysis.fit['r']))
        ax_single[0].grid(linewidth=0.2, alpha=0.7, color='black')

        ax_single[1].set_aspect(aspect=aspect_ratio)
        ax_single[1].set_ylim([ plot_range_all[col]['z'] for col in plot_range_all.columns ])
        ax_single[1].set_ylabel('Z [mm]')
        ax_single[1].set_title(f"Schnitte durch die Y-Z Ebene")
        ax_single[1].set_xlim([ plot_range_all[col]['y'] for col in plot_range_all.columns ])
        ax_single[1].set_xlabel('Y [mm]')
        ax_single[1].add_artist(plot_circle(0, analysis.fit['z'], analysis.fit['r']))
        ax_single[1].grid(linewidth=0.2, alpha=0.7, color='black')

        for n_slice in analysis.cross_section.columns.levels[0]:
            min_z_sph = float(analysis.fit['r'] - analysis.fit['z'])
            min_z = min(analysis.points['z'])
            min_z_cs = min(analysis.cross_section[n_slice]['z'])
            label = (f'{analysis.name}')
            if n_slice == '010':
                ax_single[0].scatter(analysis.cross_section[(n_slice, 'x')], analysis.cross_section[(n_slice, 'z')], s=0.1)#, label=label)
                ax_all[0].set_title(f"Schnitte durch die X-Z Ebene")
                ax_all[0].set_xlabel('X [mm]')
                ax_all[0].scatter(analysis.cross_section[(n_slice, 'x')], analysis.cross_section[(n_slice, 'z')], s=0.1, label=label)
            elif n_slice == '100':
                ax_single[1].scatter(analysis.cross_section[n_slice]['y'], analysis.cross_section[n_slice]['z'], s=0.1)#, label=label)
                ax_all[1].set_title(f"Schnitte durch die Y-Z Ebene")
                ax_all[1].set_xlabel('Y [mm]')
                ax_all[1].scatter(analysis.cross_section[(n_slice, 'y')], analysis.cross_section[(n_slice ,'z')], s=0.1, label=label)

        ax_all[1].legend(markerscale=6,
                      scatterpoints=1,
                      loc='upper center',
                      bbox_to_anchor=(0.5, -0.5),
                      fancybox=True, ncol=3)

        fig2.tight_layout()
        output_name = pathlib.Path.cwd().joinpath('output').joinpath(analysis.name + '.png')
        fig2.savefig(str(output_name), orientation='landscape', papertype='a4', dpi=600)
        plt.close(fig2)
    #-------------- all xz ------------------------------------------------------------
    fig_all.tight_layout()
    output_name = pathlib.Path.cwd().joinpath('output').joinpath('all.png')
    fig_all.savefig(str(output_name), orientation='landscape', papertype='a4', dpi=600)
    plt.close(fig_all)

def plot_contour(data):
    levels = 50
    fig = plt.figure()
    ax = fig.add_subplot(aspect=1)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f"{data.name}")
    ax.grid(linewidth=0.2, alpha=0.7, color='black')
    vmin = data.points['z'].min()
    vmax = data.points['z'].max()
    plot_val = np.linspace(vmin, vmax, levels, endpoint=True)
    cntr = ax.tricontourf(data.points['x'][::500],data.points['y'][::500],data.points['z'][::500], plot_val, vmin=vmin, vmax=vmax, extend='both')#, cmap='')
    ax.tricontour(data.points['x'][::50],data.points['y'][::50],data.points['z'][::50], levels,  linewidths=0.2, alpha=0.7, colors='black')
    cbar = fig.colorbar(cntr, ax=ax, label='Z [mm]')
    output_name = pathlib.Path.cwd().joinpath('output').joinpath(data.name + '_contour.png')
    fig.savefig(str(output_name), orientation='landscape', papertype='a4', dpi=600)
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
    return circle

def read_pkl():
    from dc_object import DataCraterAnalysis
    path = pathlib.Path.cwd()
    files = sorted([a for a in path.glob('*.pkl')])
    c_a_list = []
    for f in files:
        c_a_list.append(pickle.load(open(f, 'rb')))
    return c_a_list

def report(data):
    generate_report.generate_report(data)

if __name__ == '__main__':
    #files = sorted(glob.glob('*.stl'))
    path = pathlib.Path.cwd()
    in_dir = path.joinpath('input')
    out_dir = path.joinpath('output')
    if in_dir.exists() == False:
        in_dir.mkdir()
    if out_dir.exists() == False:
        out_dir.mkdir()
    files = sorted([a for a in in_dir.glob('*.stl')])
    crater_analysis_list = []
    for file in files:
        crater_analysis = DataCraterAnalysis()
        mesh_name = file.stem
        print(f'processing {mesh_name}')
        output_name = mesh_name + '.csv'
        mesh, mesh_points, fit_parameters = plane_fit.prepare_and_fit(str(file))
        mesh.export(out_dir.joinpath(f'{mesh_name}_transformed.stl'))
        crater_analysis.name = mesh_name
        crater_analysis.points = mesh_points
        crater_analysis.fit = fit_parameters
        for plane in [[0,1,0],[1,0,0]]:
            cs = slice_mesh(mesh, direction=plane)
            crater_analysis.cross_section = pd.concat([crater_analysis.cross_section, cs], axis=1)
            crater_analysis.cross_section.to_csv(out_dir.joinpath(f'{mesh_name}_cs.csv'), index=False)
        pickle.dump( crater_analysis, open(out_dir.joinpath(mesh_name+'.pkl'),'wb'))
        crater_analysis_list.append(crater_analysis)
        plot_contour(crater_analysis)
    plot_slices(crater_analysis_list)
    generate_report.generate_report(crater_analysis_list)

#data = pd.read_csv('file', header=[0,1])
