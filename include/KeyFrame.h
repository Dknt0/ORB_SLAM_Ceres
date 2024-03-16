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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <mutex>

#include "Frame.h"
#include "KeyFrameDatabase.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "sim3.h"

namespace ORB_SLAM2 {

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

/// @brief 关键帧
class KeyFrame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  KeyFrame(Frame& F, Map* pMap, KeyFrameDatabase* pKFDB);

  // Pose functions
  // 位姿函数，用于设置和获取关键帧的位姿

  void SetPose(const cv::Mat& Tcw);
  cv::Mat GetPose();
  cv::Mat GetPoseInverse();
  cv::Mat GetCameraCenter();
  cv::Mat GetStereoCenter();
  cv::Mat GetRotation();
  cv::Mat GetTranslation();

  // Bag of Words Representation
  // 计算词袋表示

  void ComputeBoW();

  // Covisibility graph functions
  /* 共视图 */
  // 增删改

  void AddConnection(KeyFrame* pKF, const int& weight);
  void EraseConnection(KeyFrame* pKF);
  void UpdateConnections();
  void UpdateBestCovisibles();
  // 查

  std::set<KeyFrame*> GetConnectedKeyFrames();
  std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();
  std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int& N);
  std::vector<KeyFrame*> GetCovisiblesByWeight(const int& w);
  int GetWeight(KeyFrame* pKF);

  // Spanning tree functions
  /* 生成树 */
  // 增删改

  void AddChild(KeyFrame* pKF);
  void EraseChild(KeyFrame* pKF);
  void ChangeParent(KeyFrame* pKF);
  void AddLoopEdge(KeyFrame* pKF);
  // 查

  std::set<KeyFrame*> GetChilds();
  KeyFrame* GetParent();
  bool hasChild(KeyFrame* pKF);
  std::set<KeyFrame*> GetLoopEdges();

  // MapPoint observation functions
  /* 地图点观测 */
  // 增删改

  void AddMapPoint(MapPoint* pMP, const size_t& idx);
  void EraseMapPointMatch(const size_t& idx);
  void EraseMapPointMatch(MapPoint* pMP);
  void ReplaceMapPointMatch(const size_t& idx, MapPoint* pMP);
  // 查

  std::set<MapPoint*> GetMapPoints();
  std::vector<MapPoint*> GetMapPointMatches();
  int TrackedMapPoints(const int& minObs);
  MapPoint* GetMapPoint(const size_t& idx);

  // KeyPoint functions
  /* 关键点  同 Frame */

  std::vector<size_t> GetFeaturesInArea(const float& x, const float& y,
                                        const float& r) const;
  cv::Mat UnprojectStereo(int i);

  // Image

  bool IsInImage(const float& x, const float& y) const;

  // Enable/Disable bad flag changes

  void SetNotErase();
  void SetErase();

  // Set/check bad flag

  void SetBadFlag();
  bool isBad();

  // Compute Scene Depth (q=2 median). Used in monocular.

  float ComputeSceneMedianDepth(const int q);

  /// @brief 比较权重  a>b?  适配 STL 算法
  /// @param a
  /// @param b
  /// @return
  static bool weightComp(int a, int b) { return a > b; }

  /// @brief 比较 id  pKF1<pKF2?  适配 STL 算法
  /// @param pKF1
  /// @param pKF2
  /// @return
  static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) {
    return pKF1->mnId < pKF2->mnId;
  }

  /* For ceres */
  void UpdateEigenPose();
  void UpdatePoseFromEigen();
  inline void InitOptVariable() {
    unique_lock<mutex> lock(mMutexPose);
    tcw_opt_ = tcw_;
    qcw_opt_ = qcw_;
  }
  inline void SetFromOptResult() {
    tcw_ = tcw_opt_;
    unique_lock<mutex> lock(mMutexPose);
    qcw_ = qcw_opt_;
  }
  Eigen::Vector3d tcw_;
  Eigen::Quaterniond qcw_;
  Eigen::Vector3d tcw_opt_;     // using as parameter block
  Eigen::Quaterniond qcw_opt_;  // using as parameter block
  Eigen::Vector3d tcw_gba_opt;
  Eigen::Quaterniond qcw_gba_opt;
  Eigen::Vector3d tcw_lba_opt;
  Eigen::Quaterniond qcw_lba_opt;
  Sim3 Scw_;
  Sim3 Scw_opt_;                // for pose graph optimization

  // The following variables are accesed from only 1 thread or never change (no
  // mutex needed).
 public:
  static long unsigned int nNextId;   // 下一关键帧序号
  long unsigned int mnId;             // 关键帧序号
  const long unsigned int mnFrameId;  // 帧序号
  const double mTimeStamp;            // 时间戳

  // Grid (to speed up feature matching)
  // 网格  继承自 Frame
  const int mnGridCols;                // 图像网格列数
  const int mnGridRows;                // 图像网格行数
  const float mfGridElementWidthInv;   // 网格宽度倒数
  const float mfGridElementHeightInv;  // 网格高度倒数

  // Variables used by the tracking
  // Tracking 中用到的变量
  long unsigned int
      mnTrackReferenceForFrame;  // 作为 Tracking 局部地图 KF 时，当前 F 序号
  long unsigned int
      mnFuseTargetForKF;  // LocalMapping 中进行近邻 MP 匹配融合时，当前 KF 序号
  long unsigned int mnBALocalForKF;  // 作为局部关键帧时，局部 BA 优化关键帧序号
  long unsigned int mnBALocalForKF_temp; 
  long unsigned int mnBAFixedForKF;  // 作为固定关键帧时，局部 BA 优化关键帧序号
  long unsigned int mnBAFixedForKF_temp;
  
  // Variables used by the keyframe database
  // 关键帧数据库中用到的变量
  // 用于回环候选关键帧计算，仅在调用该函数时临时使用
  // 使用这些变量时，此关键帧作为候选关键帧出现
  long unsigned int mnRelocQuery;  // 被搜索 KF id  回环候选关键帧计算
  int mnLoopWords;  // 与被搜索 KF 相同词汇数  回环候选关键帧计算
  float mLoopScore;  // 回环相似性得分  回环候选关键帧计算
  // 用于重定位候选关键帧计算，仅在调用该函数时临时使用
  // 使用这些变量时，此关键帧作为候选关键帧出现
  long unsigned int
      mnLoopQuery;  // 当前被搜索关键帧的序号  检测重定位候选关键帧计算
  int mnRelocWords;  // 与被搜索帧相同词汇的数量  检测重定位候选关键帧计算
  float mRelocScore;  // 重定位相似性得分  检测重定位候选关键帧计算

  // Variables used by the local mapping
  // Local Mapping 中用到的变量  原代码中给的是错的

  // Variables used by loop closing
  // Loop Closing 中用到的变量
  cv::Mat mTcwGBA;     // 回环存在时，用于暂存全局 BA 的结果
  cv::Mat mTcwBefGBA;  // 位姿 cw ?
  long unsigned int
      mnBAGlobalForKF;  // LoopClosing 运行全局 BA 时，当前回环 KF id

  // Calibration parameters
  // 相机内参
  const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

  // Number of KeyPoints
  const int N;  // 特征点数量

  // KeyPoints, stereo coordinate and descriptors (all associated by an index)
  /* 特征点相关  继承自 Frame */
  const std::vector<cv::KeyPoint> mvKeys;    // 左图关键点
  const std::vector<cv::KeyPoint> mvKeysUn;  // 左图去畸变关键点
  const std::vector<float>
      mvuRight;  // 特征点右图 u 坐标，双目点第三维坐标 单目取-1
  const std::vector<float> mvDepth;  // 深度 单目取-1
  const cv::Mat mDescriptors;        // 左图关键点描述子

  /* 词袋  继承自 Frame */
  DBoW2::BowVector mBowVec;  // 视觉描述向量  词袋  map<单词序号, 单词值>
  DBoW2::FeatureVector
      mFeatVec;  // 视觉特征向量  map<节点id, 关键点序号集 vector<int>>

  // Pose relative to parent (this is computed when bad flag is activated)
  cv::Mat mTcp;  // 相机系下父关键帧位姿  这个好像只会在关键帧设为坏时候被设置

  // Scale
  /* 尺度信息  继承自 Frame */
  const int mnScaleLevels;                    // 层数
  const float mfScaleFactor;                  // 层间尺度比例
  const float mfLogScaleFactor;               // 对数尺度比例
  const std::vector<float> mvScaleFactors;    // 尺度比例 >1
  const std::vector<float> mvLevelSigma2;     // 面积比例 >1
  const std::vector<float> mvInvLevelSigma2;  // 逆面积比例 <1

  // Image bounds and calibration
  /* 图像边界与内参  继承自 Frame */
  const int mnMinX;  // 去畸变后左边界
  const int mnMinY;  // 去畸变后上边界
  const int mnMaxX;  // 去畸变后右边界
  const int mnMaxY;  // 去畸变后下边界
  const cv::Mat mK;  // 内参矩阵

  // The following variables need to be accessed trough a mutex to be thread
  // safe.
 protected:
  // SE3 Pose and camera center
  cv::Mat Tcw;  // 位姿  世界相对相机
  cv::Mat Twc;  // 位姿  相机相对世界
  cv::Mat Ow;   // 平移 相机相对世界 wc ==mtwc

  // Stereo middel point. Only for visualization
  cv::Mat Cw;  // 双目相机中点

  // MapPoints associated to keypoints
  // 继承自 Frame
  std::vector<MapPoint*>
      mvpMapPoints;  // 关联到关键点的地图点
                     // 按关键点索引，未关联到地图点的关键点赋 nullptr

  // BoW
  KeyFrameDatabase* mpKeyFrameDB;  // 关键帧数据库
  ORBVocabulary* mpORBvocabulary;  // 视觉字典

  // Grid over the image to speed up feature matching
  // 继承自 Frame
  std::vector<std::vector<std::vector<size_t> > >
      mGrid;  // 网格  (x)(y)(vector<网格内特征点索引>)

  // 共视图相关
  std::map<KeyFrame*, int>
      mConnectedKeyFrameWeights;  // 共视映射 map<pKeyFrame, weight>  weight
                                  // 代表共同观测到多少个地图点 包含全部共视关系
  std::vector<KeyFrame*>
      mvpOrderedConnectedKeyFrames;  // 有序共视关键帧向量  仅包含共视大于 15 的
                                     // KF
  std::vector<int> mvOrderedWeights;  // 有序权重向量  仅包含共视大于 15 的 KF

  // Spanning Tree and Loop Edges
  // 生成树与回环边相关
  KeyFrame* mpParent;                // 生成树父关键帧节点
  std::set<KeyFrame*> mspChildrens;  // 生成树子关键帧节点集合
  std::set<KeyFrame*> mspLoopEdges;  // 回环边集合

  // Bad flags
  bool mbFirstConnection;  // 第一次被连接
  bool mbNotErase;         // 不要被清除
  bool mbToBeErased;       // 计划被清除
  bool mbBad;              // 坏关键帧

  // Only for visualization
  float mHalfBaseline;  // 基线的一半

  Map* mpMap;  // 地图

  std::mutex mMutexPose;         // 位姿锁
  std::mutex mMutexConnections;  // 共视图、生成树、回环锁
  std::mutex mMutexFeatures;     // 地图点锁
};

}  // namespace ORB_SLAM2

#endif  // KEYFRAME_H
