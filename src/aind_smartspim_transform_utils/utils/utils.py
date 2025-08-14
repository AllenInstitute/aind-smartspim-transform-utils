#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 09:59:09 2025

@author: nicholas.lusk
"""


import ants
import numpy as np
import pandas as pd


def get_orientation(params: dict) -> str:
    """
    Fetch aquisition orientation to identify origin for cell locations
    from cellfinder. Important for read_xml function in quantification
    script

    Parameters
    ----------
    params : dict
        The orientation information from processing_manifest.json

    Returns
    -------
    orient : str
        string that indicates axes order and direction current available
        options are:
            'spr'
            'sal'
        But more may be used later
    """

    orient = ["", "", ""]
    for vals in params:
        direction = vals["direction"].lower()
        dim = vals["dimension"]
        orient[dim] = direction[0]

    return "".join(orient)


def get_orientation_transform(
    orientation_in: str, orientation_out: str
) -> tuple:
    """
    Takes orientation acronyms (i.e. spr) and creates a convertion matrix for
    converting from one to another

    Parameters
    ----------
    orientation_in : str
        the current orientation of image or cells (i.e. spr)
    orientation_out : str
        the orientation that you want to convert the image or
        cells to (i.e. ras)

    Returns
    -------
    tuple
        the location of the values in the identity matrix with values
        (original, swapped)
    """

    reverse_dict = {"r": "l", "l": "r", "a": "p", "p": "a", "s": "i", "i": "s"}

    input_dict = {dim.lower(): c for c, dim in enumerate(orientation_in)}
    output_dict = {dim.lower(): c for c, dim in enumerate(orientation_out)}

    transform_matrix = np.zeros((3, 3))
    for k, v in input_dict.items():
        if k in output_dict.keys():
            transform_matrix[v, output_dict[k]] = 1
        else:
            k_reverse = reverse_dict[k]
            transform_matrix[v, output_dict[k_reverse]] = -1

    if orientation_in.lower() == "spl" or orientation_out.lower() == "spl":
        transform_matrix = abs(transform_matrix)

    original, swapped = np.where(transform_matrix.T)

    return original, swapped, transform_matrix


def calculate_scaling(
    image_res: list, downsample: int, ccf_res: int, direction: str
) -> list:
    """


    Parameters
    ----------
    image_res : list
        DESCRIPTION.
    downsample : int
        DESCRIPTION.
    ccf_res : int
        DESCRIPTION.
    direction : str
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    ds_res = [res * downsample for res in image_res]

    if direction == "forward":
        values = zip(ds_res, [ccf_res] * 3)
    elif direction == "reverse":
        values = zip([ccf_res] * 3, ds_res)

    return [res_1 / res_2 for res_1, res_2 in values]


def scale_points(points: list, scale: list) -> np.ndarray:
    """
    Takes the downsampled cells, scales and orients them in smartspim template
    space.

    Parameters
    ----------
    points : list
        list of coordinates tin a given resolution

    scale : list
        the scaling metric between the resolution of the annotation points
        and the resolution the points are being moved

    Returns
    -------
    scaled_points: np.ndarray
        list of point coordinates moved into a new resolution

    """

    scaled_points = []
    for pt in points:
        scaled_points.append(
            (pt[0] * scale[0], pt[1] * scale[1], pt[2] * scale[2])
        )

    return np.array(scaled_points)


def convert_to_ants_space(ants_parameters: dict, index_pts: np.ndarray):
    """
    Convert points from "index" space and places them into the physical space
    required for applying ants transforms for a given ANTsImage

    Parameters
    ----------
    ants_parameters : dict
        parameters of the ANTsImage physical space that you are converting
        the points
    index_pts : np.ndarray
        point coordinates in index space that have been oriented to the
        ANTs image that you are moving them into

    Returns
    -------
    ants_pts : np.ndarray
        pts converted into ANTsPy physical space

    """

    ants_pts = index_pts.copy()

    for dim in range(ants_parameters["dims"]):
        ants_pts[:, dim] *= ants_parameters["scale"][dim]
        ants_pts[:, dim] *= ants_parameters["direction"][dim]
        ants_pts[:, dim] += ants_parameters["origin"][dim]

    return ants_pts


def convert_from_ants_space(ants_parameters: dict, physical_pts: np.ndarray):
    """
    Convert points from the physical space of an ANTsImage and places
    them into the "index" space required for visualizing

    Parameters
    ----------
    template_parameters : dict
        parameters of the ANTsImage physical space from where you are
        converting the points
    physical_pts : np.ndarray
        the location of cells in physical space

    Returns
    -------
    pts : np.ndarray
        pts converted for ANTsPy physical space to "index" space

    """

    pts = physical_pts.copy()

    for dim in range(ants_parameters["dims"]):
        pts[:, dim] -= ants_parameters["origin"][dim]
        pts[:, dim] *= ants_parameters["direction"][dim]
        pts[:, dim] /= ants_parameters["scale"][dim]

    return pts


def apply_transforms_to_points(
    ants_pts: np.ndarray, transforms: list, invert: tuple
) -> np.ndarray:
    """
    Takes the cell locations that have been converted into the correct
    physical space needed for the provided transforms and registers the points

    Parameters
    ----------
    ants_pts: np.ndarray
        array with cell locations placed into ants physical space
    transforms: list
        list of the file locations for the transformations

    Returns
    -------
    transformed_pts
        list of point locations in CCF state space

    """

    df = pd.DataFrame(ants_pts, columns=["x", "y", "z"])
    transformed_pts = ants.apply_transforms_to_points(
        3, df, transforms, whichtoinvert=invert
    )

    return np.array(transformed_pts)
