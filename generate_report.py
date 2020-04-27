#!./venv/bin/python

import csv
import pathlib

import numpy as np
from jinja2 import Environment, FileSystemLoader

import pdfkit


def volume_crater(h, r):
    volume_sphere_segment = np.square(h) * np.pi / 3 * (3 * r - np.abs(h))
    return volume_sphere_segment


def highest_point_crater(points, r_crater):
    mask = np.sqrt(points["x"] ** 2 + points["y"] ** 2) < (1.3 * r_crater)
    hp_crater = max(points["z"][mask])
    return hp_crater


def generate_report(crater_analysis_list):
    path = pathlib.Path.cwd()
    out_dir = path.joinpath("output")
    if out_dir.exists() is False:
        out_dir.mkdir()
    file_loader = FileSystemLoader("templates")
    env = Environment(loader=file_loader)
    template = env.get_template("report_template.html")
    all_names = list()
    single_plots = list()
    csv_report_data = list()

    for scan in crater_analysis_list:
        cs_image = list()
        name = scan.name
        contour_image = ""
        all_names.append(name)  # für all plots
        points = scan.points
        glob_min = abs(min(points[2:]["z"]))  # warum 2: ?
        glob_max = abs(max(points[2:]["z"]))
        sph_min = abs(float((scan.fit["r"] - scan.fit["z"]) * (-1)))
        radius = abs(float(np.sqrt(scan.fit["r"] ** 2 - scan.fit["z"] ** 2) * 2))
        min_xz = abs(scan.cross_section["010"]["z"][2:].min())
        max_xz = abs(scan.cross_section["010"]["z"][2:].max())
        min_yz = abs(scan.cross_section["100"]["z"][2:].min())
        max_yz = abs(scan.cross_section["100"]["z"][2:].max())
        image_files = sorted([a for a in out_dir.glob(f"{name}*.png")])
        v_crater = volume_crater(sph_min, radius)
        max_crater = highest_point_crater(points[2:], radius)
        for img in image_files:
            if "contour" in img.name:
                contour_image = out_dir.joinpath(img)
            else:
                cs_image.append(out_dir.joinpath(img))
        # for plane in scan.cross_section:
        # scanspezifische Daten berechnen
        # -> in eine liste und später in den content
        # import ipdb; ipdb.set_trace()
        single_plots.append(
            {
                "name": name,
                "glob_min": f"{glob_min:3.2f} mm",
                "glob_max": f"{glob_max:3.2f} mm",
                "min_xz": f"{min_xz:3.2f} mm",
                "max_xz": f"{max_xz:3.2f} mm",
                "min_yz": f"{min_yz:3.2f} mm",
                "max_yz": f"{max_yz:3.2f} mm",
                "sph_min": f"{sph_min:3.2f} mm",
                "radius": f"{radius:3.2f} mm",
                "cross_sections": list(),
                "cs_image": cs_image,
                "contour_image": contour_image,
                "volume_crater": f"{v_crater:4.2f} mm^3",
                "max_crater": f"{max_crater:3.2f} mm",
            }
        )
        csv_report_data.append(
            {
                "Name": name,
                "globales Minimum": f"{glob_min:3.2f} mm",
                "tiefster Punkt Kugel": f"{sph_min:3.2f} mm",
                "tiefster Punkt XZ-Schnitt": f"{min_xz:3.2f} mm",
                "tiefster Punkt YZ-Schnitt": f"{min_yz:3.2f} mm",
                "globales Maximum": f"{glob_max:3.2f} mm",
                "höchster Punkt des Kraters": f"{max_crater:3.2f} mm",
                "höchster Punkt in XZ-Schnitt": f"{max_xz:3.2f} mm",
                "höchster Punkt in YZ-Schnitt": f"{max_yz:3.2f} mm",
                "Radius Krater (z=0)": f"{radius:3.2f} mm",
                "Volumen des Kraters (berechnet aus Kugel)": f"{v_crater:4.2f} mm^3",
            }
        )

    image_all = [a for a in out_dir.glob(f"all.png")]

    css_path = path / "templates" / "style.css"
    output = template.render(content=single_plots, image_all=image_all[0].resolve())
    pdfkit.from_string(output, out_dir.joinpath("report.pdf"), css=css_path)
    # write report-data to csv file:
    with open(f"{out_dir}/report.csv", "w", encoding="utf8", newline="") as output_file:
        fc = csv.DictWriter(output_file, fieldnames=csv_report_data[0].keys(), )
        fc.writeheader()
        fc.writerows(csv_report_data)
