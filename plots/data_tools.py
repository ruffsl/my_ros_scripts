# coding: utf-8

from __future__ import division
from __future__ import print_function

import flask
import re

from decimal import Decimal
import numpy as np
import pandas as pd
import os
# import csv

import itertools


def synchronize_Object_TF_Pose(dataDir):
    '''
    Synchronize Object and Base Pose TF

    This notebook synchronizes data from extracted bagfiles. Assuming the data directory containt are like so:
    > ```
    dataDir
        base.txt # index of pose of base by time
        object.txt # index of pose of object by time
    ```

    The resulting output is like so:

    > ```
    dataDir
        base_sync.txt # index of pose of base by time
        object_sync.txt # index of pose of object by time
    ```
    '''
    
    # ## Create paths
    base_file = os.path.join(dataDir,'base.txt')
    object_file = os.path.join(dataDir,'object.txt')

    base_sync_file = os.path.join(dataDir,'base_sync.txt')
    object_sync_file = os.path.join(dataDir,'object_sync.txt')
    
    # ## Load pose info into dataframe
    base_df = pd.read_csv(base_file, header=None, delim_whitespace=True, dtype=str)
    object_df = pd.read_csv(object_file, header=None, delim_whitespace=True, dtype=str)
    
    # ## Load and associate data
    base_array = np.array(base_df[0], dtype=np.dtype(Decimal))
    object_array = np.array(object_df[0], dtype=np.dtype(Decimal))
    
    idx = np.searchsorted(base_array, object_array) - 1
    mask = idx >= 0
    
    # ## Save pose file as well
    base_df2 = base_df.ix[idx]
    base_df2.to_csv(base_sync_file, header=None, index=False)

    object_df2 = object_df[mask]
    object_df2.to_csv(object_sync_file, header=None, index=False)
    
    print('\tbase_df2: ', base_df2.shape)
    print('\tobject_df2: ', object_df2.shape)

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0
    
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def rowToPose(row):
    pose = quaternion_matrix([row[4], row[5], row[6], row[7]])
    pose[:,3] = np.array([row[1], row[2], row[3], 1])
    return np.matrix(pose)
    

def getGlobalErrors (base_poses_path, object_poses_path, object_error_path, start_index=0):
    '''return object error from base and object files
    Calucates pose error of tacked object using csv files of TF's from base frame and object
    '''
    
    object_df = pd.read_csv(base_poses_path, header=None)
    object_df = object_df.set_index(0)
    
    base_df = pd.read_csv(object_poses_path, header=None)
    base_df = base_df.set_index(0)

    errors = np.array([])
    
    base_zero = rowToPose(base_df.iloc[start_index])
    object_zero = base_zero*rowToPose(object_df.iloc[start_index])
    iterzip = itertools.izip(base_df.iterrows(), object_df.iterrows())
    
    i = 0
    for [base_index, base_row], [object_index, object_row] in iterzip:
        if (i < start_index):
            continue
        else:
            base_pose = rowToPose(base_row)
            object_pose = base_pose*rowToPose(object_row)
            diff = object_zero - object_pose
            error = np.linalg.norm(diff[:,0:2])
            errors = np.append(errors, [error])

    errors = pd.Series(errors)
    errors.to_csv(object_error_path, header=None, index=False)
    
# def getFrameErrors (base_poses_path, object_poses_path, object_error_path, start_index=0):
#     '''return object error from base and object files
#     Calucates pose error of tacked object using csv files of TF's from base frame and object
#     '''
    
#     object_df = pd.read_csv(base_poses_path, header=None)
#     object_df = object_df.set_index(0)
    
#     base_df = pd.read_csv(object_poses_path, header=None)
#     base_df = base_df.set_index(0)

#     errors = np.array([])
    
#     vo = rowToPose(base_df.iloc[start_index])
#     vo[:,3] = np.array([0, 0, 0, 1])
    
#     object_zero = base_zero*rowToPose(object_df.iloc[start_index])
#     iterzip = itertools.izip(base_df.iterrows(), object_df.iterrows())
    
#     i = 0
#     for [base_index, base_row], [object_index, object_row] in iterzip:
#         if (i < start_index):
#             continue
#         else:
#             vo1 = rowToPose(base_row)
#             vo1[:,3] = np.array([0, 0, 0, 1])
            
#             vo1 = np.linalg.inv(vo)*vo1
#             vo1_temp_R = np.identity((4,4))
#             vo1_temp_R[:2,:2] = vo1[:2,:2]
            
#             diff_T = np.zeros((4,4))
#             diff_T[:,3] = object_pose(:,3) - vo1(:,3)
            
#             object_pose_temp = np.identity((4,4))
#             object_pose_temp[:2,:2] = object_pose[:2,:2]            
            
#             ob_pose = vo1_temp_R*(object_pose_temp + diff_T);
            
#             object_pose = rowToPose(object_row)
            
#             ob_poses = [ob_poses;ob_pose];
            
            
            
            
#             error = np.linalg.inv(obj_poses)*ob_pose;
            
#             error = np.linalg.norm(error[:,3])
#             errors = np.append(errors, [error])

#     errors = pd.Series(errors)
#     errors.to_csv(object_error_path, header=None, index=False)
    
# for i =start:size(poses,1)
#     vo1=[];
#     ob_pose = [];
    
#     vo1 = [quaternion2matrix([poses(i,7),poses(i,4:6)]),poses(i,1:3)'];
#     vo1 = [vo1;0,0,0,1];  
#     vo_all = [vo_all;vo1];
    
#     vo1 = inv(vo)*vo1;
#     ob_pose = [vo1(1:3,1:3),[0;0;0;];0,0,0,1]*([object_pose(1:3,1:3),object_pose(1:3,4)-vo1(1:3,4);0,0,0,1]);
#     %ob_pose = [ob_pose(1:3,1:3),[0;0;0];0,0,0,1]'*ob_pose;
    
#     obj_poses = [quaternion2matrix([object_poses(i,7),object_poses(i,4:6)]),object_poses(i,1:3)'];
#     obj_poses = [obj_poses;0,0,0,1];

    
#     ob_poses = [ob_poses;ob_pose];
#     error = inv(obj_poses)*ob_pose;
# %    error = obj_poses(1:3,4)-ob_pose(1:3,4);
#     errors(i,1) = norm(error(1:3,4));
# end

app = flask.Flask(__name__)

LATEX_SUBS = (
    (re.compile(r'\\'), r'\\textbackslash'),
    (re.compile(r'([{}_#%&$])'), r'\\\1'),
    (re.compile(r'~'), r'\~{}'),
    (re.compile(r'\^'), r'\^{}'),
    (re.compile(r'"'), r"''"),
    (re.compile(r'\.\.\.+'), r'\\ldots'),
)

def escape_tex(value):
    newval = value
    for pattern, replacement in LATEX_SUBS:
        newval = pattern.sub(replacement, newval)
    return newval

texenv = app.create_jinja_environment()
texenv.block_start_string = '((*'
texenv.block_end_string = '*))'
texenv.variable_start_string = '((('
texenv.variable_end_string = ')))'
texenv.comment_start_string = '((='
texenv.comment_end_string = '=))'
texenv.filters['escape_tex'] = escape_tex