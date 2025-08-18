#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:24:18 2025

@author: nicholas.lusk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:21:46 2025

@author: nicholas.lusk
"""

import os
import ants
from glob import glob

import numpy as np
import pandas as pd

from aind_smartspim_transform_utils import base_dir
from aind_smartspim_transform_utils.io import file_io as fio
from aind_smartspim_transform_utils.utils import utils


def _check_path(name: str, sub_folder: str):  # pragma: no cover
    fpath = os.path.join(base_dir, name, sub_folder)

    if not os.path.exists(fpath):
        os.makedirs(fpath)
        return False

    return True


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
        file_check = _check_path(name, "transforms")

        data_folder = "SmartSPIM-template_2024-05-16_11-26-14"
        files = [
            "spim_template_to_ccf_syn_0GenericAffine_25.mat",
            "spim_template_to_ccf_syn_1Warp_25.nii.gz",
            "spim_template_to_ccf_syn_1InverseWarp_25.nii.gz",
        ]
        dest = os.path.join(base_dir, name.lower(), "transforms")

        if not file_check:
            fio._download_data_from_s3(data_folder, files, dest)

        root = os.path.join(base_dir, name, "transforms")

        transforms["ccf_from_image"] = [
            glob(os.path.join(root, "*.mat"))[0],
            glob(os.path.join(root, "*1InverseWarp_25.nii.gz"))[0],
        ]

        transforms["image_to_ccf"] = [
            glob(os.path.join(root, "*1Warp_25.nii.gz"))[0],
            glob(os.path.join(root, "*.mat"))[0],
        ]
    else:
        raise ValueError(
            f"name: {name} is not a currently available transformation"
        )

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
        file_check = _check_path(name, "ccf")

        data_folder = "SmartSPIM-template_2024-05-16_11-26-14"
        files = ["ccf_average_template_25.nii.gz"]
        dest = os.path.join(base_dir, name.lower(), "ccf")

        if not file_check:
            fio._download_data_from_s3(data_folder, files, dest)

        root = os.path.join(base_dir, name, "ccf")

        ants_ccf, ccf_info = fio.load_ants_nifti(
            f"{root}/ccf_average_template_25.nii.gz"
        )
    else:
        raise ValueError(f"name: {name} is not a currently available ccf")

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
        file_check = _check_path(name, "template")

        data_folder = "SmartSPIM-template_2024-05-16_11-26-14"
        files = ["smartspim_lca_template_25.nii.gz"]
        dest = os.path.join(base_dir, name.lower(), "template")

        if not file_check:
            fio._download_data_from_s3(data_folder, files, dest)

        root = os.path.join(base_dir, name, "template")

        ants_template, template_info = fio.load_ants_nifti(
            f"{root}/smartspim_lca_template_25.nii.gz"
        )
    else:
        raise ValueError(f"name: {name} is not a currently available ccf")

    return ants_template, template_info


def _fetch_zarr_data(  # pragma: no cover
    dataset_path: str, channel: str, level: int
) -> list:
    zarr_path = os.path.join(
        dataset_path,
        "image_tile_fusing/OMEZarr",
        channel + ".zarr",
        str(level),
        ".zarray",
    )

    return fio._read_json_as_dict(zarr_path)


def _get_estimated_downsample(
    voxel_resolution: list,
    registration_res: tuple = (16.0, 14.4, 14.4),
) -> int:
    """
    Get the estimated multiscale based on the provided
    voxel resolution. This is used for image stitching.

    e.g., if the original resolution is (1.8. 1.8, 2.0)
    in XYZ order, and you provide (3.6, 3.6, 4.0) as
    image resolution, then the picked resolution will be
    1.

    Parameters
    ----------
    voxel_resolution: List[float]
        Image original resolution. This would be the resolution
        in the multiscale "0".
    registration_res: Tuple[float]
        Approximated resolution that was used for registration
        in the computation of the transforms. Default: (16.0, 14.4, 14.4)
    """

    downsample_versions = []
    for idx in range(len(voxel_resolution)):
        downsample_versions.append(
            registration_res[idx] // float(voxel_resolution[idx])
        )

    downsample_res = int(min(downsample_versions))
    return round(np.log2(downsample_res))


def _parse_acquisition_data(acquisition_dict: dict):
    """
    Retrieves the relevant imaging information from the acquisition.json
    that is required for transforming points


    Parameters
    ----------
    acquisition_dict : dict
        The data from loading the acquisition.json

    Returns
    -------
    orientation : dict
        Dictionary containing the axes names (i.e. X, Y, Z), the imaging
        resolution of each axis, the dimension order of the axes, and the
        direction of each axis

    """

    orientation = acquisition_dict["axes"]

    scales = {}
    for scale, axis in zip(
        acquisition_dict["tiles"][0]["coordinate_transformations"][1]["scale"],
        ["X", "Y", "Z"],
    ):
        scales[axis] = scale

    for c, axis in enumerate(orientation):
        for s, res in scales.items():
            if s == axis["name"]:
                axis["resolution"] = res
                orientation[c] = axis

    channels = []

    for tile in acquisition_dict["tiles"]:
        channel = tile["file_name"].split("/")[0]
        if channel not in channels:
            channels.append(channel)

    acquisition = {
        "orientation": orientation,
        "registration": _get_estimated_downsample(
            [s[1] for s in sorted(scales.items(), reverse=True)]
        ),
        "channels": channels,
    }

    return acquisition


def rename_transforms(transforms: dict) -> dict:
    """
    renames transforms to make sense with images. Kind of silly we do this but
    overall I think helpful for people

    Parameters
    ----------
    transforms : str
        location of the transforms and acquisition.json for a given dataset
        if there is no acquisition.json will only register to template

    Returns
    -------

    transforms: dict
        same transforms just renamed 

    """

    rename_map = {"points_to_ccf": "ccf_from_image", "points_from_ccf": "image_to_ccf"} 
    transforms = {rename_map.get(k, k): v for k, v in transforms.items()}

    return transforms


class ImageTransform:
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

    acquisition: dict
        metadata for your dataset loaded from the acquisition.json

    image_metadata: dict
    """

    def __init__(  # pragma: no cover
        self,
        name: str,
        dataset_transforms: list,
        acquisition: dict,
        image_metadata: dict,
    ):
        self.ccf_transforms = _get_ccf_transforms(name)
        self.ccf_template, self.ccf_template_info = _get_ccf_template(name)
        self.ls_template, self.ls_template_info = _get_ls_template(name)

        self.dataset_transforms = rename_transforms(dataset_transforms)
        self.acquisition = _parse_acquisition_data(acquisition)
        self.zarr_shape = image_metadata["shape"]
        self.template_orientation = {
                "anterior_to_posterior": 1,
                "superior_to_inferior": 2,
                "right_to_left": 0,
            }

    def forward_transform(
        self,
        image: np.array,
        ccf_res=25,
        reg_ds = None
    ) -> np.array:
        """
        Moves a 3D image from raw image space into CCF space

        Parameters
        ----------
        image : np.array
            A 3D volume in raw image space
        ccf_res: int
            The resolution of the ccf used in registration

        Returns
        -------
        registered_iamge : np.array
            3D volume registered into CCF space

        """
        
        # flip axis based on the template orientation relative to input image
        img_array = image.astype(np.double)
        
        img_out, in_mat, out_mat = utils.check_orientation(
            img_array,
            self.acquisition["orientation"],
            self.template_orientation,
        )
        
        print(f"Image has been oriented to template: {img_out.shape}")
        
        spacing_order = np.where(in_mat)[1]
        
        if reg_ds is None:
            print(f'No downsample factor is given. Using factor {self.acquisition["registration"]} based on acquisition')
            reg_ds = self.acquisition["registration"]
        else:
            print(f"Downsample factor of {reg_ds} provided for registration")
            
        spacing = [0 ,0, 0]
        for o in self.acquisition['orientation']:
            spacing[o['dimension']] = o['resolution'] * 2**reg_ds
        
        img_spacing = tuple([spacing[s] for s in spacing_order])
        
        ants_img = ants.from_numpy(img_out, spacing=img_spacing)
        ants_img.set_direction(self.ls_template.direction)
        ants_img.set_origin(self.ls_template.origin)
            
        print('################################')
        print(f"Ants Image: {ants_img}")
        print(f"Light Sheet Template: {self.ls_template}")
        print(f"CCF Template: {self.ccf_template}")
        print('################################')

        # apply transform to template
        aligned_image = ants.apply_transforms(
            fixed=self.ls_template,
            moving=ants_img,
            transformlist=self.dataset_transforms['image_to_ccf'],
        )

        aligned_image = ants.apply_transforms(
            fixed=self.ccf_template,
            moving=aligned_image,
            transformlist=self.ccf_transforms['image_to_ccf'],
        )

        return aligned_image.numpy()

    def reverse_transform(
        self,
        dataset_image: np.array,
        image: np.array,
        ccf_res=25,
        reg_ds = None
    ) -> np.array:
        """
        Moves points from CCFv3 space into light sheet state space.

        Parameters
        ----------
        points : pd.DataFrame
            array of points in CCFv3 space. Input columns for dataframe must
            have column names defined as 'ML', 'AP', and 'DV'
        ccf_res: int
            The resolution of the ccf used in registration

        Returns
        -------
        transformed_pts : np.array
            array of points in light sheet space
        """
        
        img_array = image.astype(np.double)
        img_spacing = tuple([ccf_res] * 3)
        
        ants_img = ants.from_numpy(img_array, spacing=img_spacing)
        ants_img.set_direction(self.ccf_template.direction)
        ants_img.set_origin(self.ccf_template.origin)

        # apply transform to template
        aligned_image = ants.apply_transforms(
            fixed=self.ls_template,
            moving=ants_img,
            transformlist=self.ccf_transforms['image_to_ccf'],
        )
        
        if reg_ds is None:
            print(f'No downsample factor is given. Using factor {self.acquisition["registration"]} based on acquisition')
            reg_ds = self.acquisition["registration"]
        else:
            print(f"Downsample factor of {reg_ds} provided for registration")
        
        dataset_array = dataset_image.astype(np.double)
        
        img_out, in_mat, out_mat = utils.check_orientation(
            dataset_array,
            self.acquisition["orientation"],
            self.template_orientation,
        )
        
        spacing_order = np.where(in_mat)[1]
        
        spacing = [0 ,0, 0]
        for o in self.acquisition['orientation']:
            spacing[o['dimension']] = o['resolution'] * 2**reg_ds
        
        img_spacing = tuple([spacing[s] for s in spacing_order])
        
        ants_dataset = ants.from_numpy(img_out, spacing=img_spacing)


        aligned_image = ants.apply_transforms(
            fixed=ants_dataset,
            moving=aligned_image,
            transformlist=self.ls_transforms['image_to_ccf'],
        )
        
        
        return aligned_image.numpy()