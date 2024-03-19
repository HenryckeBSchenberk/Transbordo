import numpy as np
import cv2
from cv2 import aruco
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
board = aruco.CharucoBoard((7, 5), 1, .8, dictionary)

def show_board(board):
    workdir = "./workdir/"
    # board.generateImage()
    imboard = board.generateImage((2000, 2000))
    cv2.imwrite(workdir + "chessboard.tiff", imboard)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")
    plt.show()

def read_chessboards(images, aruco_dict, board):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize

def read_chess_frames(datadir = "../../data/calib_tel_ludo/"):
    images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png") ])
    # order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    # images = images[order]
    return images

def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

if __name__ == "__main__":
    import pickle
    images =read_chess_frames("./board/")
    allCorners,allIds,imsize = read_chessboards(images, dictionary, board)
    ret, mtx, dist, rvect, tvecs = calibrate_camera(allCorners, allIds, imsize)
    # pickle.dumps((mtx, dist),  open("mtx_dist.p", "wb"))

    h,  w = imsize
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)

    mapx, mapy = cv2.convertMaps(map1=mapx, map2=mapy, dstmap1type=cv2.CV_16SC2)
    # print(mapx)
    pickle.dump((mapx, mapy),  open("mp_xy.pkl", "wb"))
    

    # img = cv2.imread("./board/Image__2024-02-23__18-57-06.png")
    for path in images:
        img = cv2.imread(path)
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        cv2.imshow("frame", dst)
        cv2.imshow("frame2", dst2)
        cv2.waitKey(1)

    # print(allCorners)