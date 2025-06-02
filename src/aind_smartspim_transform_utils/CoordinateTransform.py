#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:21:46 2025

@author: nicholas.lusk
"""

import os
import numpy as np
import dask as da

from glob import glob

from .aind_smartspim_transform_utils import file_io as fio
from .aind_smartspim_transform_utils import utils

def _parse_acquisition_data(data_description: dict):
    
    orientation = data_description['prelim_acquisition']['axes']
    resolution = data_description['pipeline_processing']['resolution']
    
    for c, axis in enumerate(orientation):
        for res in resolution:
            if res['axis_name'] == axis['name']:
                axis['resolution'] = res['resolution']
                orientation[c] = axis
                
    return orientation


def _get_ccf_transforms(name: str) -> dict:
    """
    Loads static transforms that move points between the CCFv3 and 
    SmartSPIM-LCA template.

    Parameters
    ----------
    name : str
        name of the template you want to load. Currently 'SmartSPIM-LCA'
        is the only option

    Returns
    -------
    transforms: dict
        the transforms needed for moving pts forward or backward

    """
    
    transforms = {}
    
    if name == "SmartSPIM-LCA":
        transforms['points_to_ccf'] = [
            glob(os.path.join('path/to/affine', 'affine.mat'))[0],
            glob(os.path.join('path/to/inverse_warp', 'inverse_warp.nii.gz'))[0]
        ]
        
        transforms['points_from_ccf'] = [
            glob(os.path.join('path/to/warp', 'warp.nii.gz'))[0],
            glob(os.path.join('path/to/affine', 'affine.mat'))[0]
        ]
    else:
        ValueError(f"name: {name} is not a currently available transformation")
        

    return transforms



def get_dataset_transforms(manifest_path: str) -> dict:
    """
    Loads the dynamic transforms for a given dataset. dataset path can either
    be a local location or the S3 bucket location for a given dataset

    Parameters
    ----------
    manifest_path : str
        location of the transforms and acquisition.json for a given dataset
        if there is no acquisition.json will only register to template

    Returns
    -------
    acquisition: dict
        The acquisition parameters for registering from template to raw space
        if it is not provided will be None
        
    transforms: dict
        the transforms needed for moving pts forward or backward

    """
    
    transforms = {}
    
    if not os.path.exists(manifest_path):
        FileExistsError(f"{manifest_path} does not exist.")
        
        
    manifest, transforms = fio.get_transforms(manifest_path)
        
    
    return manifest, transforms

    
class CoordinateTransform():
    """
    Class for transforming pts between light sheet and CCFv3 space
    """
    
    def __init__(
            self, 
            name: str, 
            dataset_transforms: list,
            processing_manifest: dict
    ):
        
        self.ccf_transforms = _get_ccf_transforms(name)
        self.ccf_template = _get_ccf_template(name)
        self.ls_template = _get_ls_template(name)
        
        self.dataset_transforms = dataset_transforms
        self.acquisition = _parse_acquisition_data(processing_manifest)
        

    def forward_transform(
        self, 
        coordinates: np.array,
        input_img: da.array,
        registration_ds: int
    ) -> np.array:
        """
        Moves points from light sheet state space into CCFv3 space
        
        Parameters
        ----------
        coordinates : np.array
            array of points in raw light sheet space
        reg_ds : int
            The level of downsampling that was done during registration. The 
            default for SmartSPIM is 3
            
        Returns
        -------
        transformed_pts : np.array
            array of points in CCFv3 space

        """
        
        # Getting downsample res
        ds = 2**reg_ds
        reg_dims = [dim / ds for dim in input_res]

        
        # get orientation information
        orient = utils.get_orientation(self.orientation)
        template_params = utils.get_template_info(image_files["smartspim_template"])
        
        _, swapped, mat = utils.get_orientation_transform(
            orient, template_params["orientation"]
        )
        
        
        scaled_cells = utils.scale_points(self.coordinates, scaling)
        orient_cells = scaled_cells[:, swapped]
    
        template_params = utils.get_template_info(image_files["smartspim_template"])
        ants_pts = utils.convert_to_ants_space(template_params, orient_cells)
        
        template_pts = utils.apply_transforms_to_points(
            ants_pts, 
            self.dataset_transforms['points_to_ccf'], 
            invert=(True, False)
        )
        
        ccf_pts = utils.apply_transforms_to_points(
            template_pts, 
            self.ccf_transforms['points_to_ccf'], 
            invert=(True, False)
        )
        
        ccf_params = utils.get_template_info(image_files["ccf_template"])
        ccf_pts = utils.convert_from_ants_space(ccf_params, ccf_pts)
        
        _, swapped, _ = utils.get_orientation_transform(
            template_params["orientation"], ccf_params["orientation"]
        )
        
        transformed_pts = ccf_pts[:, swapped]
        
    
        return transformed_pts

    def reverse_transform(
            self, 
            coordinates: np.array,
            reg_ds: int
    ) -> np.array:
        """
        Moves points from CCFv3 space into light sheet state space

        Parameters
        ----------
        coordinates : np.array
            array of points in CCFv3 space
        reg_ds : int
            The level of downsampling that was done during registration. The 
            default for SmartSPIM is 3

        Returns
        -------
        transformed_pts : np.array
            array of points in light sheet space
        """
        
        # Getting downsample res
        ds = 2**reg_ds
        reg_dims = [dim / ds for dim in input_res]

        
        # get orientation information
        orient = utils.get_orientation(self.orientation)
        template_params = utils.get_template_info(image_files["smartspim_template"])
        
        _, swapped, mat = utils.get_orientation_transform(
            orient, template_params["orientation"]
        )
        
        
        scaled_cells = utils.scale_points(self.coordinates, scaling)
        orient_cells = scaled_cells[:, swapped]
    
        template_params = utils.get_template_info(image_files["smartspim_template"])
        ants_pts = utils.convert_to_ants_space(template_params, orient_cells)
        
        template_pts = utils.apply_transforms_to_points(
            ants_pts, 
            self.dataset_transforms['points_from_ccf'], 
            invert=(False, False)
        )
        
        ccf_pts = utils.apply_transforms_to_points(
            template_pts, 
            self.ccf_transforms['points_from_ccf'], 
            invert=(False, False)
        )
        
        ccf_params = utils.get_template_info(image_files["ccf_template"])
        ccf_pts = utils.convert_from_ants_space(ccf_params, ccf_pts)
        
        _, swapped, _ = utils.get_orientation_transform(
            template_params["orientation"], ccf_params["orientation"]
        )
        
        transformed_pts = ccf_pts[:, swapped]
    
    
        return transformed_pts