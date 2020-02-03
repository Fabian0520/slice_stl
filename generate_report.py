#!./venv/bin/python

from jinja2 import Environment, FileSystemLoader
import pdfkit
import os
import glob
import numpy as np

def generate_report(crater_analysis_list):
    cwd = os.getcwd() + '/'
    file_loader = FileSystemLoader('templates')
    env = Environment(loader=file_loader)
    template = env.get_template('about.html')

    content = list()

    for scan in crater_analysis_list:
        cs_image=list()
        name = scan.name
        # Bilder sortieiren und in content
        points = scan.points
        glob_min = abs(min(points[2:]['z']))
        glob_max = abs(max(points[2:]['z']))
        sph_min = abs(float((scan.fit['r'] - scan.fit['z']) * (-1)))
        radius = abs(float(np.sqrt(scan.fit['r']**2 - scan.fit['z']**2)*2))
        min_xz = abs(scan.cross_section[0]['z'][2:].min())
        min_yz = abs(scan.cross_section[1]['z'][2:].min())
        image_files = glob.glob(f'{name}*.png')
        for img in image_files:
            if 'contour' in img:
                contour_image = cwd + img
            else:
                cs_image.append(cwd + img)
        #for plane in scan.cross_section:
            # scanspezifische Daten berechnen
            # -> in eine liste und später in den content
        #import ipdb; ipdb.set_trace()
        content.append({'name' : name,
                        'glob_min' : f'{glob_min:3.2f} mm',
                        'glob_max' : f'{glob_max:3.2f} mm',
                        'min_xz' : f'{min_xz:3.2f} mm',
                        'min_yz' : f'{min_yz:3.2f} mm',
                        'sph_min' : f'{sph_min:3.2f} mm',
                        'radius' : f'{radius:3.2f} mm',
                        'cross_sections' : list(),
                        'cs_image' : cs_image,
                        'contour_image' : contour_image
                       })
    output = template.render(content=content)
    pdfkit.from_string(output, 'test.pdf', css=cwd+'templates/style.css')
