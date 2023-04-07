# import the necessary packages
import numpy as np
import imutils
import cv2
class Stitcher:
    def __init__(self):
        # 确定是否使用OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)
        
    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
            showMatches=False):
            # 解包图像，然后检测关键点并提取
            # 定位局部不变描述子
            (imageB, imageA) = images
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            # 匹配两个图像之间的特征
            M = self.matchKeypoints(kpsA, kpsB,
                featuresA, featuresB, ratio, reprojThresh)
            #如果匹配为None，则没有足够的匹配
            # 创建全景图的关键点
            if M is None:
                return None
            # 否则，应用透视变形来缝合图像
            # 连接
            (matches, H, status) = M
            result = cv2.warpPerspective(imageA, H,
                (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
            # 检查匹配的关键点是否可以可视化
            if showMatches:
                vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                    status)
                # 返回拼接后的图像和
                return (result, vis)
            # 返回拼接后的图像
            return result
    def detectAndDescribe(self, image):
        # 将图像转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检查是否使用OpenCV 3.X
        if self.isv3:
            # 检测并提取图像特征
            descriptor = cv2.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        # 否则，使用OpenCV 2.4.X
        else:
            #检测图像中的关键点
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            # 从图像中提取特征
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
        # 将KeyPoint对象中的关键点转换为NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # 返回关键点和特征的元组
        return (kps, features)
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # 计算原始匹配，并初始化实际匹配列表
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # 计算原始匹配，并初始化实际匹配列表
        for m in rawMatches:
            # 确保距离在一定比例范围内 (即Lowe的比例测试)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                
        # 计算单应性矩阵至少需要4个匹配
        if len(matches) > 4:    
            # 构造两组点
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # 计算两组点之间的单应性矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                              reprojThresh)
            # 返回匹配及其对应的单应性矩阵和匹配点的状态
            return (matches, H, status)
        # 否则，无法计算单应性矩阵
        return None
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化输出可视化图像
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # 遍历匹配
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 仅处理成功匹配的关键点
            if s == 1:
                # 绘制匹配线
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # 返回可视化结果
        return vis
    
 
# imageA = cv2.imread('2.png')
# imageB = cv2.imread('1.png')
imageA = cv2.imread('C.jpg')
imageB = cv2.imread('D.jpg')

imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
#将图像拼接在一起创建全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)

