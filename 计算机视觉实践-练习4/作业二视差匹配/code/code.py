import numpy as np
import cv2
import sys

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == '__main__':

        # Read left and right images
        imgL = cv2.imread('Adirondack_left.png', cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread('Adirondack_right.png', cv2.IMREAD_GRAYSCALE)    
        # We need grayscale for disparity map.
#         gray_left = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#         gray_right = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        disparity_image = depth_map(imgL, imgR)  # Get the disparity map

        # Show the images
        # cv2.imshow('left(R)', imgL)
        # cv2.imshow('right(R)', imgR)
        # cv2.imshow('Disparity', disparity_image)
        cv2.imwrite('Adirondack_disparity.jpg', disparity_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
#             break
            cv2.destroyAllWindows()