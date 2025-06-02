#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 14:20:43 2025

@author: nicholas.lusk
"""

import os
from aind_smartspim_transform_utils.io import file_io as fio


def check_layer_info(ng_data, layer_name):
    """
    Checks the provided layer name and make sure it contains the appropriate
    data for coordinate transforms.
        
        name: must be a string
        type: annotation
        contents: list of points
        

    Parameters
    ----------
    layer_name : str
        name of neuroglancer layer  containing points to transform
        
    Returns
    -------
    None.

    """
    
    if not isinstance(layer_name, str):
        TypeError(f"Layer name must be <type 'str'>'. Provided type {type(layer_name)}")
        
    
    
    
    

def read_neuroglancer_json(filepath: str, layer_name: str) -> dict:
    """
    Takes a neuroglancer JSON and retrieves coordinates from a
    annotation layer

    Parameters
    ----------
    filepath : str
        DESCRIPTION.
    layer_name : str
        DESCRIPTION.

    Returns
    -------
    coordinates : TYPE
        DESCRIPTION.

    """
    
    if not os.path.exists(filepath):
        FileNotFoundError(f"{filepath} does not exist")
    
    
    neuroglancer_data = fio.read_json_as_dict(filepath)
    
    layer_names = [layers['name'] for layers in neuroglancer_data['layers']] 

    if layer_name in layer_names:
        cells = []
        for layers in neuroglancer_data['layers']:
            if layers['name'] == layer_name:
                for cell in layers['annotations']:
                    cells.append(
                        [
                            cell['point'][0],
                            cell['point'][1],
                            cell['point'][2]
                        ]
                    )
    else:
        ValueError(f"There is no layer named {layer_name} in neuroglancer data")
        
    return neuroglancer_data
        
def get_neuroglancer_annotation_points(
    neuroglancer_data: dict, 
    layer_name: str,
    spacing = None
) -> list:
    
    """
    Takes points from a neuroglancer annotation layer and returns them
    as a numpy array. The coordinates are ordered [z, y, x]
    
    Parameters
    ----------
    
    neuroglancer_data: dict
        A dictionary of the neuroglancer state imported from a JSON
        
    layer_name: str
        The name of the layer that you are retrieving points
        
    spacing: List[float] Optional
        A list of floats that specify the resolutiuon unit of the points.
        By default the points correspond to voxel location so no spacing is 
        needed. Default: None
    
    
    layer_names = [layers['name'] for layers in neuroglancer_data['layers']] 

    if layer_name not in layer_names:
        ValueError(f"There is no layer named {layer_name} in neuroglancer data")
        

    points = []
    annotations = layer.get("annotations", [])
    for annotation in annotations:
        point_arr = np.array(annotation.get("point", []), dtype=float)
        if point_arr.shape[0] != 4:
            raise ValueError(
                "Annotation points expected to have 4 dimensions "
                f"(z, y, x, t), but {point_arr.shape[0]} found."
            )
        points.append(point_arr[:3])  # Keep only the first three dimensions
    points = np.stack(points) if points else np.empty((0, 3), dtype=float)
    if spacing is not None:
        points = points * spacing

    if return_description:
        descriptions = [
            annotation.get("description", None) for annotation in annotations
        ]
        return points, np.array(descriptions, dtype=object)
    return points, None

def get_neuroglancer_metadata(neuroglancer_data: dict) -> dict:
    """
    Retrive necessary data from neuroglancer JSON for transforming
    points
    """

    Parameters
    ----------
    neuroglancer_data : dict
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """