###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    WW = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]])
    Q1 = u.dot(WW.dot(vh))
    Q2 = u.dot(WW.T.dot(vh))
    R1 = np.linalg.det(Q1)*Q1
    R2 = np.linalg.det(Q2)*Q2
    # np.concatenate([x, y], axis=1)
    u12, u3 = np.hsplit(u, [2])
    RT1 = np.concatenate([R1, u3], axis=1)
    RT2 = np.concatenate([R1, -u3], axis=1)
    RT3 = np.concatenate([R2, u3], axis=1)
    RT4 = np.concatenate([R2, -u3], axis=1)
    # print(RT1)
    RT = np.array([RT1, RT2, RT3, RT4])
    # print(np.shape(RT))
    # print(RT)
    # ans = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
    #     [0.1019, 0.9948, 0.0045, -0.0089],
    #     [0.2041, -0.0254, 0.9786, 0.0331]])
    # RT = [ans, ans, ans, ans]
    return RT
    raise Exception('Not Implemented Error')

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    MM = len(image_points)
    DLT = np.empty([2*MM, 4])
    b = np.array([[0], [0], [0], [0]])
    for x in range(MM):
        u = image_points[x][0]
        v = image_points[x][1]
        # print("CM = ", camera_matrices[x])
        M1, M2, M3 = np.vsplit(camera_matrices[x], [1, 2])
        temp1 = np.array(v).dot(M3) - M2
        temp2 = M1 - np.array(u).dot(M3)
        DLT[x*2] = temp1
        DLT[x*2 + 1] = temp2

    #DLT = np.array(DLT)
    # print(np.shape(DLT))
    u, s, vh = np.linalg.svd(DLT, full_matrices=True)
    # print("vh = ", np.shape(vh)[0])
    # print("M = ", MM)
    P1, P2 = np.hsplit(vh.T,[np.shape(vh)[0] - 1])
    # print("P2 = ", np.shape(P2))
    #P = np.array([P2[0][0]/P2[3][0], P2[1][0]/P2[3][0], P2[2][0]/P2[3][0]])
    P = np.array([P2[0][0], P2[1][0], P2[2][0]])/P2[3][0]
    # print(P)
    return P
    raise Exception('Not Implemented Error')

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    MM = np.shape(image_points)[0]
    e = np.array([0, 0])
    P = np.concatenate([point_3d,[1]])
    P = P.reshape((4, 1))
    # print(P)
    for x in range(MM):
        y = camera_matrices[x].dot(P)
        # print("CM = ", camera_matrices)
        # print("P = ", P)
        # print("y = ", y)
        p = [y[0][0]/y[2][0], y[1][0]/y[2][0]]
        # print("p = ", p)
        e = np.hstack((e, p - image_points[x]))
    e1, e = np.hsplit(e,[2])
    # print("e = ", e)
    return e
    raise Exception('Not Implemented Error')

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    MM = np.shape(camera_matrices)[0]
    P = np.concatenate([point_3d,[1]])
    P = P.reshape((4, 1))
    J = np.array([0, 0, 0])
    for x in range(MM):
        MiP = camera_matrices[x].dot(P)
        M1 = np.array([camera_matrices[x][0][0], camera_matrices[x][0][1], camera_matrices[x][0][2]])
        M2 = np.array([camera_matrices[x][1][0], camera_matrices[x][1][1], camera_matrices[x][1][2]])
        M3 = np.array([camera_matrices[x][2][0], camera_matrices[x][2][1], camera_matrices[x][2][2]])
        # e1 = M1/MiP[2] - ((M3*MiP[0])/MiP[2])/MiP[2]
        # e2 = M2/MiP[2] - ((M3*MiP[1])/MiP[2])/MiP[2]
        e1 = ((M1*MiP[2] - M3*MiP[0])/MiP[2]**2)
        e2 = ((M2*MiP[2] - M3*MiP[1])/MiP[2]**2)
        J = np.vstack((J, e1))
        J = np.vstack((J, e2))
    J1, J = np.vsplit(J,[1])
    return J
    raise Exception('Not Implemented Error')

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    P = linear_estimate_3d_point(image_points, camera_matrices)
    # e = reprojection_error(P, image_points, camera_matrices)
    # J = jacobian(P, camera_matrices)
    # JT = J.T
    # JTJinverse = np.linalg.inv(JT.dot(J))
    # print(P)
    for x in range(10):
        # print("iteration = ", x+1)
        e = reprojection_error(P, image_points, camera_matrices)
        # print("e = ", e)
        J = jacobian(P, camera_matrices)
        JT = J.T
        JTJ = JT.dot(J)
        #print(J)
        # JTJinverse = np.linalg.inv(JTJ)
        try:
            JTJinverse = np.linalg.inv(JTJ)
        except:
            print("inverse error in iteration ", x + 1)
            break
        JTe = JT.dot(e)
        P = P - JTJinverse.dot(JTe)
    # print("P = ", P)
    return P
    raise Exception('Not Implemented Error')

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    RT4 = estimate_initial_RT(E)
    M4 = [K.dot(RT4[0]), K.dot(RT4[1]), K.dot(RT4[2]), K.dot(RT4[3])]
    # print(np.shape(M4))
    # print(M4)
    I = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    positive_number = np.array([0, 0, 0, 0])
    for x in range(4):
        for y in range(np.shape(image_points)[0]):
            temp = nonlinear_estimate_3d_point(image_points[y], np.array([I, RT4[x]]))
            P = np.concatenate([temp,[1]]).T
            if RT4[x].dot(P)[2] > 0 and P[2] > 0:
                positive_number[x] = positive_number[x] + 1
            #print(np.shape(nonlinear_estimate_3d_point(image_points[y], np.array([I, RT4[x]]))))
    # print(positive_number)
    index = np.argmax(positive_number)
    return RT4[index]
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')
