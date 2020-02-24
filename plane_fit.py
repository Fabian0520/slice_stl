import trimesh
import numpy as np
import pandas as pd


def prepare_and_fit(mesh_file):
    mesh = trimesh.load_mesh(mesh_file)
    # -----------------------------------------------------------------------------------------------------------------
    #               Fit Plane
    plane_fit = trimesh.points.plane_fit(mesh.vertices)
    trans_matrix = trimesh.geometry.plane_transform(plane_fit[0], plane_fit[1])
    mesh.apply_transform(trans_matrix)
    # convert vertices to pd.DF
    mesh_points = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])
    # calculate histogram
    hist_values, hist_bins = np.histogram(mesh_points["z"], bins=100, density=True)
    # import ipdb; ipdb.set_trace()
    # find sharpest edge in histogram and take this as zero plane
    null_pkt = np.array(
        [
            np.absolute(hist_values[i - 1] - hist_values[i])
            for i in range(1, len(hist_values))
        ]
    ).argmax()
    # import ipdb; ipdb.set_trace()
    # mask points so they are near hist_max
    mask = mesh_points["z"].between(hist_bins[null_pkt], hist_bins[null_pkt + 1])
    # new fit and transform
    plane_fit = trimesh.points.plane_fit(mesh.vertices[mask])
    trans_matrix = trimesh.geometry.plane_transform(plane_fit[0], plane_fit[1])
    mesh.apply_transform(trans_matrix)

    mesh_points = pd.DataFrame(
        mesh.vertices, columns=["x", "y", "z"]
    )  # generate new points, after transform
    mask = mesh_points["z"].between(-1, 1)
    plane_fit = trimesh.points.plane_fit(mesh.vertices[mask])
    trans_matrix = trimesh.geometry.plane_transform(plane_fit[0], plane_fit[1])
    mesh.apply_transform(trans_matrix)
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    #                                       fit sphere to points not in plane
    # generate new points, after transform
    mesh_points = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])
    # take only points with z<-1.0 *** wert noch automatisch eryeugen
    mask = mesh_points["z"] < -1.0
    sph_fit_c, sph_fit_r, sph_fit_err = trimesh.nsphere.fit_nsphere(mesh_points[mask])
    # Maske "Abstand Punkte von Kugel kleiner x" erzeugen und noch mal fitten (x10)
    for i in range(0, 10):
        # Distance between points and sphere surface
        mesh_points["dist_sph"] = np.absolute(
            np.linalg.norm(mesh_points.loc[:, "x":"z"] - sph_fit_c, axis=1.0)
            - sph_fit_r
        )
        mask = mesh_points["dist_sph"] < 0.5  # 1.0 - (i/60)
        sph_fit_c, sph_fit_r, sph_fit_err = trimesh.nsphere.fit_nsphere(
            mesh_points.loc[:, "x":"z"][mask]
        )
    # import ipdb; ipdb.set_trace()

    # aligning sphere and mesh
    # trimesh.transformations.compose_matrix()
    # align mesh and sphere in xy plane, so that origin is colinear with c_sphere
    trans_matrix_mesh = np.array(
        [[1, 0, 0, -sph_fit_c[0]], [0, 1, 0, -sph_fit_c[1]], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    mesh.apply_transform(trans_matrix_mesh)
    sph_fit_c[:2] = 0  # x und y of sphere are 0, since the mesh has been moved
    mesh_points = pd.DataFrame(
        mesh.vertices, columns=["x", "y", "z"]
    )  # generate new points, after transform
    fit_parameters = pd.DataFrame([sph_fit_c[2]], columns=["z"])
    fit_parameters["r"] = sph_fit_r
    fit_parameters["error"] = sph_fit_err
    # -----------------------------------------------------------------------------------------------------------------

    return mesh, mesh_points, fit_parameters
