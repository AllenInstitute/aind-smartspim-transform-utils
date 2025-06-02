#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 13:59:08 2025

@author: nicholas.lusk
"""

import os
import json
import ants

import numpy as np

from glob import glob


def _check_transforms(transforms_list: list):
    """
    There are many versions of registration. This check informs the user if
    the dataset was directly registered or if it used template-based

    Parameters
    ----------
    transforms_list : list
        DESCRIPTION.

    Returns
    -------
    None.

    """
    

def get_transforms(transforms_path: str, data_description_path: str):
    """
    Get the paths to the transforms for the provided dataset as well as the
    required acquisition information for transforming points

    Parameters
    ----------
    transform_path : str
        path to the transforms created during registration
        
    data_description_path : str
        path to the data_description file created after stitching

    Returns
    -------
    data_description : dict
        dictionary containing acquisition metadata required for transforming
        points
    transforms : list
        list of all transform files from registration

    """
    
    data_description_file = os.path.join(data_description_path, 'data_description.json')
    
    if not os.path.isfile(data_description_path):
        FileNotFoundError(f"data description file cannot be found at: {data_description_path}")
        
    data_description = read_json_as_dict(data_description_file)
    
    
    transforms = {}
    
    try:
        transforms['pts_to_ccf'] = [
                glob(os.path.join(transforms_path, 'SyN_0GenericAffine.mat'))[0],
                glob(os.path.join(transforms_path, 'InverseWarp.nii.gz'))[0],
        ]
    except:
        FileNotFoundError("Could not find files needed for moving points from light sheet to CCF")
        
    try:
        transforms['pts_from_ccf'] = [
            glob(os.path.join(transforms_path, 'SyN_1Warp.nii.gz'))[0],
            glob(os.path.join(transforms_path, 'SyN_0GenericAffine.mat'))[0],
        ]
    except:
        FileNotFoundError("Could not find files needed for moving points from CCF to light sheet")
        
    return data_description, transforms      


def read_json_as_dict(filepath: str) -> dict:
    """
    Loads json file

    Parameters
    ----------
    filepath : str
        path to json file

    Returns
    -------
    data: dict
        loaded json formated as a dict

    """
    
    if not os.path.exists(filepath):
        FileNotFoundError(f"File {filepath} does not exist.")
    
    
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    
    return data

def load_ants_nifti(filepath: str) -> dict:
    """
    Loads an ants image object and returns image information
    

    Parameters
    ----------
    filepath : str
        location of the ants nii.gz file

    Returns
    -------
    image: ants.image
        ants object
        
    description: dict
        dictionary with descriptive information related to the ants image

    """
    
    if not os.path.exists(filepath):
        FileNotFoundError(f"File {filepath} does not exist.")
    
    ants_img = ants.image_read(filepath)

    description = {
        "orientation": ants_img.orientation,
        "dims": ants_img.dimension,
        "scale": ants_img.spacing,
        "origin": ants_img.origin,
        "direction": ants_img.direction[np.where(ants_img.direction != 0)],
    }
    
    return ants_img, description
    
    
    
    