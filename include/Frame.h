/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

namespace ORB_SLAM2 {
#define FRAME_GRID_ROWS 48  // 图像网格行数
#define FRAME_GRID_COLS 64  // 图像网格列数

class MapPoint;
class KeyFrame;

class Frame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  /* 构造函数 */

  Frame();
  Frame(const Frame &frame);
  Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
        ORBextractor *extractorLeft, ORBextractor *extractorRight,
        ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
        const float &thDepth);
  Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
        ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K,
        cv::Mat &distCoef, const float &bf, const float &thDepth);
  Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor,
        ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
        const float &thDepth);

  // Extract ORB on the image. 0 for left image and 1 for right image.

  void ExtractORB(int flag, const cv::Mat &im);

  // Compute Bag of Words representation.

  void ComputeBoW();

  // Set the camera pose.

  void SetPose(cv::Mat Tcw);

  // Computes rotation, translation and camera center matrices from the camera
  // pose.

  void UpdatePoseMatrices();

  // Returns the camera center.

  /// @brief 获取相机相机相对世界位置 two
  /// @return
  inline cv::Mat GetCameraCenter() { return mOw.clone(); }

  // Returns inverse of rotation

  /// @brief 获取 Rwc
  /// @return
  inline cv::Mat GetRotationInverse() { return mRwc.clone(); }

  // Check if a MapPoint is in the frustum of the camera
  // and fill variables of the MapPoint to be used by the tracking

  bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

  // Compute the cell of a keypoint (return false if outside the grid)

  bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

  vector<size_t> GetFeaturesInArea(const float &x, const float &y,
                                   const float &r, const int minLevel = -1,
                                   const int maxLevel = -1) const;

  // Search a match for each keypoint in the left image to a keypoint in the
  // right image. If there is a match, depth is computed and the right
  // coordinate associated to the left keypoint is stored.

  void ComputeStereoMatches();

  // Associate a "right" coordinate to a keypoint if there is valid depth in the
  // depthmap.

  void ComputeStereoFromRGBD(const cv::Mat &imDepth);

  // Backprojects a keypoint (if stereo/depth info available) into 3D world
  // coordinates.

  cv::Mat UnprojectStereo(const int &i);

 public:
  /* For ceres */
  void UpdateEigenPose();
  void UpdatePoseFromEigen();
  inline void InitOptVariable() {
    tcw_opt_ = tcw_;
    qcw_opt_ = qcw_;
  }
  inline void SetFromOptResult() {
    tcw_ = tcw_opt_;
    qcw_ = qcw_opt_;
  }
  Eigen::Vector3d tcw_;
  Eigen::Quaterniond qcw_;
  Eigen::Vector3d tcw_opt_;     // using as parameter block
  Eigen::Quaterniond qcw_opt_;  // using as parameter block

  /* 外部工具 */
  // Vocabulary used for relocalization.
  ORBVocabulary *mpORBvocabulary;  // 视觉字典
  // Feature extractor. The right is used only in the stereo case.
  ORBextractor *mpORBextractorLeft, *mpORBextractorRight;  // 特征提取器

  /* 位置与时间 */
  // Current and Next Frame id.
  static long unsigned int nNextId;  // 下一帧 id
  long unsigned int mnId;            // 帧 id
  // Camera pose.
  cv::Mat mTcw;  // 相机位姿
  // Frame timestamp.
  double mTimeStamp;  // 时间戳
  // Reference Keyframe.
  KeyFrame *mpReferenceKF;  // 参考关键帧指针  Tracking 中用到
  static bool mbInitialComputations;  // 初始化标志位

  /* 相机参数 */
  // Calibration matrix and OpenCV distortion parameters.
  cv::Mat mK;  // 内参矩阵
  // 以下都是静态变量，因此只能有一种相机   注意长度应该以像素为单位
  static float fx;
  static float fy;
  static float cx;
  static float cy;
  static float invfx;
  static float invfy;
  cv::Mat mDistCoef;  // 畸变参数
  // Stereo baseline multiplied by fx.
  float mbf;  // 双目基线 * fx
  // Stereo baseline in meters.
  float mb;  // 双目基线  最小深度  米
  // Threshold close/far points. Close points are inserted from 1 view.
  // Far points are inserted as in the monocular case from 2 views.
  float mThDepth;  // 远近双目点阈值

  /* 特征相关 */
  // Number of KeyPoints.
  int N;  // 左图关键点数量
  // Keypoints are assigned to cells in a grid to reduce matching complexity
  // when projecting MapPoints. 特征点分配到网格
  static float mfGridElementWidthInv;   // 网格宽度倒数
  static float mfGridElementHeightInv;  // 网格高度倒数
  std::vector<std::size_t>
      mGrid[FRAME_GRID_COLS]
           [FRAME_GRID_ROWS];  // 去畸变左图网格 (x)(y)(vector<关键点序号>)
  // Vector of keypoints (original for visualization) and undistorted (actually
  // used by the system). In the stereo case, mvKeysUn is redundant as images
  // must be rectified. In the RGB-D case, RGB images can be distorted.
  std::vector<cv::KeyPoint> mvKeys;       // 左图关键点
  std::vector<cv::KeyPoint> mvKeysRight;  // 右图关键点
  std::vector<cv::KeyPoint> mvKeysUn;     // 左图去畸变关键点
  // Corresponding stereo coordinate and depth for each keypoint.
  // "Monocular" keypoints have a negative value.
  std::vector<float> mvuRight;  // 特征点右图 u 坐标，双目点第三维坐标  单目取-1
  std::vector<float> mvDepth;  // 深度  单目取-1
  // ORB descriptor, each row associated to a keypoint.
  cv::Mat mDescriptors, mDescriptorsRight;  // ORB 描述子
  // MapPoints associated to keypoints, NULL pointer if no association.
  std::vector<MapPoint *> mvpMapPoints;  // 关联到特征点的地图点
  // Flag to identify outlier associations.
  std::vector<bool> mvbOutlier;  // 关键点外点标志位  仅在存在 MP 观测时使用  在
                                 // PoseOptimization 使用
  // Undistorted Image Bounds (computed once).
  // 去畸变图像边界
  static float mnMinX;  // 去畸变后左边界
  static float mnMaxX;  // 去畸变后右边界
  static float mnMinY;  // 去畸变后上边界
  static float mnMaxY;  // 去畸变后下边界

  /* 词袋相关 */
  // Bag of Words Vector structures.
  DBoW2::BowVector mBowVec;  // 视觉单词向量  map<单词序号, 单词值>
  DBoW2::FeatureVector
      mFeatVec;  // 视觉特征向量  map<节点id, 关键点序号集 vector<int>>

  /* 特征提取器参数 */
  // Scale pyramid info.
  // 从左提取器获得
  int mnScaleLevels;                // 层数
  float mfScaleFactor;              // 尺度比例
  float mfLogScaleFactor;           // 对数尺度比例
  vector<float> mvScaleFactors;     // 尺度比例 >1
  vector<float> mvInvScaleFactors;  // 逆尺度比例 <1
  vector<float> mvLevelSigma2;      // 面积比例  >1
  vector<float> mvInvLevelSigma2;   // 逆面积比例  <1

 private:
  // Undistort keypoints given OpenCV distortion parameters.
  // Only for the RGB-D case. Stereo must be already rectified!
  // (called in the constructor).

  void UndistortKeyPoints();

  // Computes image bounds for the undistorted image (called in the
  // constructor).

  void ComputeImageBounds(const cv::Mat &imLeft);

  // Assign keypoints to the grid for speed up feature matching (called in the
  // constructor).

  void AssignFeaturesToGrid();

  // Rotation, translation and camera center
  cv::Mat mRcw;  // 旋转矩阵 世界相对相机 cw
  cv::Mat mtcw;  // 平移 世界相对相机 cw
  cv::Mat mRwc;  // 旋转矩阵 相机相对世界 wc
  cv::Mat mOw;   // 平移 相机相对世界 wc ==mtwc
};

}  // namespace ORB_SLAM2

#endif  // FRAME_H
