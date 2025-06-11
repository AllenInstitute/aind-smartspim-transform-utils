#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:21:46 2025

@author: nicholas.lusk
"""

import os
import boto3
import numpy as np
import dask.array as da

from glob import glob
from tqdm import tqdm

from aind_smartspim_transform_utils import base_dir
from aind_smartspim_transform_utils.io import file_io as fio
from aind_smartspim_transform_utils.utils import utils

def _get_client():
    '''
    Creates an instance of a s3 client for interacting with AWS

    Returns
    -------
    client.s3
        S3 object of botocore.client module

    '''
    return boto3.client('s3')

def _check_path(name: str, sub_folder: str):
    
    fpath = os.path.join(base_dir, name, sub_folder)
    
    if not os.path.exists(fpath):
        os.makedirs(fpath)
        return False
    
    return True   

def _download_data(name: str, subfolder: str):
    
    dest = os.path.join(base_dir, name, subfolder)
    client = _get_client()
    
    if name.lower() == "smartspim_lca":
        
        data_folder = 'SmartSPIM-template_2024-05-16_11-26-14'
        
        if subfolder == 'ccf':
            files = ['ccf_average_template_25.nii.gz']
        elif subfolder == 'template':
            files = ['smartspim_lca_template_25.nii.gz']
        elif subfolder == 'transforms':
            files = [
                'spim_template_to_ccf_syn_0GenericAffine_25.mat',
                'spim_template_to_ccf_syn_1Warp_25.nii.gz',
                'spim_template_to_ccf_syn_1InverseWarp_25.nii.gz'
            ]
            
        for file in files:
            fname = os.path.join(dest, file)
            s3_object_key = f'{data_folder}/{file}'
            
            meta_data = client.head_object(Bucket='aind-open-data', Key=s3_object_key)
            total_len = int(meta_data.get('ContentLength', 0))
            bar = "{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}"
            
            with tqdm(
                    total = total_len, 
                    desc = s3_object_key, 
                    bar_format=bar, 
                    unit='B', 
                    unit_scale = True
                ) as pbar:
                
                with open(fname, 'wb') as f:
                    client.download_fileobj(
                        'aind-open-data', 
                        s3_object_key, 
                        f, 
                        Callback=pbar.update
                    )
    
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
    
    if name.lower() == "smartspim_lca":
        
        file_check = _check_path(name, 'transforms')
        
        if not file_check:
            _download_data(name, 'transforms')
        
        root = os.path.join(base_dir, name, 'transforms')
        
        transforms['points_to_ccf'] = [
            glob(os.path.join(root, '*.mat'))[0],
            glob(os.path.join(root, '*1InverseWarp_25.nii.gz'))[0]
        ]
        
        transforms['points_from_ccf'] = [
            glob(os.path.join(root, '*1Warp_25.nii.gz'))[0],
            glob(os.path.join(root, '*.mat'))[0]
        ]
    else:
        ValueError(f"name: {name} is not a currently available transformation")
        

    return transforms

def _get_ccf_template(name):
    """
    loads the nifti file for the ccf you need based on the nape you provide
    to the transfrom function. If you currently do not have that file it will
    download it from S3

    Parameters
    ----------
    name : str
        The name of the transform process you are using

    Returns
    -------
    ants_ccf : ants.array
        The template volume loaded as an ants object
    ccf_info : dict
        dictionary describing the template. Includes the orientation, axes 
        directions, dimensions, scale, and origin of the ants.array

    """
    
    if name.lower() == "smartspim_lca":
        
        file_check = _check_path(name, 'ccf')
        
        if not file_check:
            _download_data(name, 'ccf')
        
        root = os.path.join(base_dir, name, 'ccf')
        
        ants_ccf, ccf_info = fio.load_ants_nifti(f"{root}/ccf_average_template_25.nii.gz")
    else:
        ValueError(f"name: {name} is not a currently available ccf")  
    
    
    return ants_ccf, ccf_info

def _get_ls_template(name):
    """
    loads the nifti file for the lightsheet template you need based on the 
    name you provide to the transfrom function. If you currently do not have 
    that file it will download it from S3

    Parameters
    ----------
    name : str
        The name of the transform process you are using

    Returns
    -------
    ants_template: ants.array
        The template volume loaded as an ants object
    template_info : dict
        dictionary describing the template. Includes the orientation, axes 
        directions, dimensions, scale, and origin of the ants.array
    """

    
    if name.lower() == "smartspim_lca":
        
        file_check = _check_path(name, 'template')
        
        if not file_check:
            _download_data(name, 'template')
        
        root = os.path.join(base_dir, name, 'template')
        
        
        ants_template, template_info = fio.load_ants_nifti(f"{root}/smartspim_lca_template_25.nii.gz")
    else:
        ValueError(f"name: {name} is not a currently available ccf")  
    
    return ants_template, template_info

def _parse_acquisition_data(manifest: dict):
    """
    Retrieves the relevant imaging information from the processing manifest
    that is required for transforming points
    

    Parameters
    ----------
    manifest : dict
        The data from loading the processing manifest

    Returns
    -------
    orientation : dict
        Dictionary containing the axes names (i.e. X, Y, Z), the imaging
        resolution of each axis, the dimension order of the axes, and the
        direction of each axis

    """
    
    orientation = manifest['prelim_acquisition']['axes']
    resolution = manifest['pipeline_processing']['stitching']['resolution']
    
    for c, axis in enumerate(orientation):
        for res in resolution:
            if res['axis_name'] == axis['name']:
                axis['resolution'] = res['resolution']
                orientation[c] = axis
                
    return orientation


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
    
    Currently there is only one option for name
        - smartspim_lca
        
    Parameters
    ----------
    
    name: str
        The name of the transforms that you want to use
        
    dataset_transforms: list
        A list of the dataset specific transforms you want to use
        
    processing_manifest: dict
        metadata for your dataset loaded from the processing_manifest.json
    """
    
    def __init__(
            self, 
            name: str, 
            dataset_transforms: list,
            processing_manifest: dict
    ):
        
        self.ccf_transforms = _get_ccf_transforms(name)
        self.ccf_template, self.ccf_template_info = _get_ccf_template(name)
        self.ls_template, self.ls_template_info = _get_ls_template(name)
        
        self.dataset_transforms = dataset_transforms
        self.acquisition = _parse_acquisition_data(processing_manifest)
        

    def forward_transform(
        self, 
        points: np.array,
        input_image: da.Array,
        image_res: list,
        reg_ds: int,
        ccf_res = 25,
    ) -> np.array:
        """
        Moves points from light sheet state space into CCFv3 space
        
        Parameters
        ----------
        coordinates : np.array
            array of points in raw light sheet space
        input_image: da.array
            dask array of the image that the points were annotated on
        reg_ds : int
            The level of downsampling that was done during registration. The 
            default for SmartSPIM is 3
        ccf_res: int
            The resolution of the ccf used in registration
            
        Returns
        -------
        transformed_pts : np.array
            array of points in CCFv3 space

        """
        
        # downsample points to registration resolution
        points_ds = points / 2**reg_ds
        
        # get dimensions of registered image for orienting points
        input_shape = input_image.shape
        if len(input_shape) == 5:
            input_shape = input_shape[2:]
            

        reg_dims = [dim / 2**reg_ds for dim in input_shape]
        
        # flip axis based on the template orientation relative to input image
        orient = utils.get_orientation(self.acquisition)
        
        _, swapped, mat = utils.get_orientation_transform(
            orient, self.template_info["orientation"]
        )
        
        for idx, dim_orient in enumerate(mat.sum(axis=1)):
            if dim_orient < 0:
                points_ds[:, idx] = reg_dims[idx] - points_ds[:, idx]
        
        
        image_res = [dim['resolution'] for dim in self.acquisition]
        
        #scale points and orient axes to template
        scaling = utils.calculate_scaling(
            image_res = image_res,
            downsample = 2**reg_ds,
            ccf_res = ccf_res,
            direction = 'forward'
        )
        
        scaled_pts = utils.scale_points(points_ds, scaling)
        orient_pts = scaled_pts[:, swapped]
        
        # convert points into ccf space
        ants_pts = utils.convert_to_ants_space(self.template_info, orient_pts)
        
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
        

        ccf_pts = utils.convert_from_ants_space(self.ccf_info, ccf_pts)
        
        _, swapped, _ = utils.get_orientation_transform(
            self.template_info["orientation"], self.ccf_info["orientation"]
        )
        
        transformed_pts = ccf_pts[:, swapped]
        
    
        return transformed_pts

    def reverse_transform(
            self, 
            points: np.array,
            input_image: da.Array,
            image_res: list,
            reg_ds: int,
            ccf_res = 25,
    ) -> np.array:
        """
        Moves points from CCFv3 space into light sheet state space.

        Parameters
        ----------
        points : np.array
            array of points in CCFv3 space
        reg_ds : int
            The level of downsampling that was done during registration. The 
            default for SmartSPIM is 3

        Returns
        -------
        transformed_pts : np.array
            array of points in light sheet space
        """
        
        #TODO make this so it is not hard coded
        # orient points from CCF visual to CCF ants orientation
        ccf_pts = points[:, [0, 2, 1]]
        
        
        # convert points into raw space
        ants_pts = utils.convert_to_ants_space(self.ccf_info, ccf_pts)
        
        template_pts = utils.apply_transforms_to_points(
            ants_pts, 
            self.ccf_transforms['points_from_ccf'], 
            invert=(False, False)
        )
        
        raw_pts = utils.apply_transforms_to_points(
            template_pts, 
            self.dataset_transforms['points_from_ccf'], 
            invert=(False, False)
        )
        
        raw_pts = utils.convert_from_ants_space(self.template_info, raw_pts)
        
        
        # get dimensions of registered image for orienting points
        input_shape = input_image.shape
        if len(input_shape) == 5:
            input_shape = input_shape[2:]
            
        reg_dims = [dim / 2**reg_ds for dim in input_shape]
        
        # flip axis based on the template orientation relative to input image
        orient = utils.get_orientation(self.orientation)
        
        _, swapped, mat = utils.get_orientation_transform(
            self.template_info["orientation"], orient
        )
        
        for idx, dim_orient in enumerate(mat.sum(axis=1)):
            if dim_orient < 0:
                raw_pts[:, idx] = reg_dims[idx] - raw_pts[:, idx]
        
        
        #scale points and orient axes to original image
        scaling = utils.calculate_scaling(
            image_res = image_res,
            downsample = 2**reg_ds,
            ccf_res = ccf_res,
            direction = 'reverse'
        )
        
        scaled_pts = utils.scale_points(raw_pts, scaling)
        orient_pts = scaled_pts[:, swapped]
        
        # upsample points from registration to raw image space
        transformed_pts = orient_pts / 2**reg_ds
    
        return transformed_pts