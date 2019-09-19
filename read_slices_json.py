import pandas as pd
import glob, os
import matplotlib.pyplot as plt

def read_slice():
    data = dict()
    for file in glob.glob('*.json'):
        name = file.split('.')[0]
        data[name] = pd.read_json(file)
    return data

def set_x_to_zero(input_data):
    data = input_data
    for slice_number in data.keys():
        delta_loc = data[slice_number]['z'].idxmin()
        delta_val = data[slice_number]['x'][delta_loc]
        if delta_val > 0:
            data[slice_number]['x'] = data[slice_number]['x'] - abs(delta_val)
        else:
            data[slice_number]['x'] = data[slice_number]['x'] + abs(delta_val)
    return data

def plot_all_slices(data, aspect_ratio = 1):
    fig, ax = plt.subplots(figsize=(11.6929, 8.26772))   # Din A4 Größe in inch
    ax.set_aspect(aspect = aspect_ratio)
    ax.set_title('Schnitte durch den tiefsten Punkt')
    ax.set_ylabel('Z [mm]')
    ax.set_xlabel('X [mm]')
    for slice_number in data.keys():
        ax.scatter(data[slice_number]['x'], data[slice_number]['z'],
                   label=(slice_number + '\n'r'min_z = '+'{:6.2f}'.format(data[slice_number]['z'].min())), s=0.5)
    ax.legend(markerscale=6, scatterpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.5),fancybox=True, ncol=3)
    # weiteres Argument: ncol=?
    ax.grid()
    fig.tight_layout()
    fig.show()
    fig.savefig('output.png', orientation='landscape', papertype='a4', dpi=600)

