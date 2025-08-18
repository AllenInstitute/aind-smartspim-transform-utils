#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 09:59:09 2025

@author: nicholas.lusk
"""


import ants
import numpy as np
import pandas as pd

def rotate_image(img: np.array, in_mat: np.array, reverse: bool):
    """
    Rotates axes of a volume based on orientation matrix.

    Parameters
    ----------
    img: np.array
        Image volume to be rotated
    in_mat: np.array
        3x3 matrix with cols indicating order of input array and rows
        indicating location to rotate axes into

    Returns
    -------
    img_out: np.array
        Image after being rotated into new orientation
    out_mat: np.array
        axes correspondance after rotating array. Should always be an
        identity matrix
    reverse: bool
        if you are doing forward or reverse registration

    """

    if not reverse:
        in_mat = in_mat.T

    original, swapped = np.where(in_mat)
    img_out = np.moveaxis(img, original, swapped)

    out_mat = in_mat[:, swapped]
    for c, row in enumerate(in_mat):
        val = np.where(row)[0][0]
        if row[val] == -1:
            img_out = np.flip(img_out, c)
            out_mat[val, val] *= -1

    return img_out, out_mat


def check_orientation(img: np.array, params: dict, orientations: dict):
    """
    Checks aquisition orientation an makes sure it is aligned to the CCF. The
    CCF orientation is:
        - superior_to_inferior
        - left_to_right
        - anterior_to_posterior

    Parameters
    ----------
    img : np.array
        The raw image in its aquired orientation
    params : dict
        The orientation information from processing_manifest.json
    orientations: dict
        The axis order of the CCF reference atals

    Returns
    -------
    img_out : np.array
        The raw image oriented to the CCF
    """

    orient_mat = np.zeros((3, 3))
    acronym = ["", "", ""]

    for k, vals in enumerate(params):
        direction = vals["direction"].lower()
        dim = vals["dimension"]
        if direction in orientations.keys():
            ref_axis = orientations[direction]
            orient_mat[dim, ref_axis] = 1
            acronym[dim] = direction[0]
        else:
            direction_flip = "_".join(direction.split("_")[::-1])
            ref_axis = orientations[direction_flip]
            orient_mat[dim, ref_axis] = -1
            acronym[dim] = direction[0]

    # check because there was a bug that allowed for invalid spl orientation
    # all vals should be postitive so just taking absolute value of matrix
    if "".join(acronym) == "spl":
        orient_mat = abs(orient_mat)

    img_out, out_mat = rotate_image(img, orient_mat, False)

    return img_out, orient_mat, out_mat


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
