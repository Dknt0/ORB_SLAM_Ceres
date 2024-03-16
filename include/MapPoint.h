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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>
#include <opencv2/core/core.hpp>

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"

namespace ORB_SLAM2 {

class KeyFrame;
class Map;
class Frame;

/// @brief 地图点类
class MapPoint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  /* 构造函数 */

  MapPoint(const cv::Mat& Pos, KeyFrame* pRefKF, Map* pMap);
  MapPoint(const cv::Mat& Pos, Map* pMap, Frame* pFrame, const int& idxF);

  /* 位置 */

  void SetWorldPos(const cv::Mat& Pos);
  cv::Mat GetWorldPos();

  /* 观测相关 */
  // 功能

  void AddObservation(KeyFrame* pKF, size_t idx);
  void EraseObservation(KeyFrame* pKF);
  void Replace(MapPoint* pMP);
  void ComputeDistinctiveDescriptors();
  void UpdateNormalAndDepth();
  void IncreaseVisible(int n = 1);
  void IncreaseFound(int n = 1);

  // 查找查询

  std::map<KeyFrame*, size_t> GetObservations();
  int Observations();
  int GetIndexInKeyFrame(KeyFrame* pKF);
  bool IsInKeyFrame(KeyFrame* pKF);
  cv::Mat GetNormal();
  KeyFrame* GetReferenceKeyFrame();
  float GetFoundRatio();
  inline int GetFound() { return mnFound; }
  MapPoint* GetReplaced();
  cv::Mat GetDescriptor();
  float GetMinDistanceInvariance();
  float GetMaxDistanceInvariance();

  // 标志

  void SetBadFlag();
  bool isBad();

  // 尺度预测

  int PredictScale(const float& currentDist, KeyFrame* pKF);
  int PredictScale(const float& currentDist, Frame* pF);

 public:
  /* For ceres */
  inline void UpdateEigenPose() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    for (size_t i = 0; i < 3; ++i) {
      twp_[i] = mWorldPos.at<float>(i, 0);
    }
  }
  inline void UpdatePoseFromEigen() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    for (size_t i = 0; i < 3; ++i) {
      mWorldPos.at<float>(i, 0) = twp_[i];
    }
  }
  inline void InitOptVariable() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    twp_opt_ = twp_;
  }
  inline void SetFromOptResult() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    twp_ = twp_opt_;
  }
  Eigen::Vector3d twp_;
  Eigen::Vector3d twp_opt_;  // using as parameter block
  Eigen::Vector3d twp_gba_opt;
  Eigen::Vector3d twp_lba_opt;
  unsigned int idx_lba_;
  


  /* 序号 */
  long unsigned int mnId;            // 地图点序号
  static long unsigned int nNextId;  // 下一个地图点序号，静态变量
  long int mnFirstKFid;  // 第一次观测到这个点的关键帧序号
  long int mnFirstFrame;  // 第一次观测到这个点的帧序号，可能是关键帧的帧序号
  int nObs;  // 关键帧观测次数   双目点(包括深度点)算两次观测 其他点算一次观测

  // Variables used by the tracking
  /* Tracking 中用到的变量 */
  /// Frame::isInFrustum 用到
  float mTrackProjX;      // 投影到当前帧 u
  float mTrackProjY;      // 投影到当前帧 v
  float mTrackProjXR;     // 投影到当前帧 右图 u
  bool mbTrackInView;     // 对于当前帧可视
  int mnTrackScaleLevel;  // 金字塔尺度预测级别
  float mTrackViewCos;    // 视角余弦
  ///
  long unsigned int
      mnTrackReferenceForFrame;  // Tracking 中作为局部 MP 时，当前 F 序号
  long unsigned int mnLastFrameSeen;  // 最后一次被观测 帧索引

  // Variables used by local mapping
  /* LocalMapping 中用到的变量 */
  long unsigned int mnBALocalForKF;        // 局部 BA 优化关键帧序号
  long unsigned int mnBALocalForKF_temp;   // for test
  long unsigned int mnFuseCandidateForKF;  // LocalMapping 中进行近邻 MP
                                           // 匹配融合时，当前 KF 序号

  // Variables used by loop closing
  /* LoopClosing 中用到的变量 */
  long unsigned int mnLoopPointForKF;  // LoopClosing 中作为回环匹配 KF 附近局部
                                       // MP 时，当前 KF 序号
  long unsigned int mnCorrectedByKF;  // LoopClosing 中作为当前 KF 局部 MP
                                      // 进行位置修正时，当前 KF 序号
  long unsigned int mnCorrectedReference;  // LoopClosing 中作为当前 KF 局部 MP
                                           // 进行位置修正时，修正参考 KF 序号
  KeyFrame* mnCorrectedReferencePtr = nullptr;
  cv::Mat mPosGBA;  // 回环存在时，用于暂存全局 BA 的结果
  long unsigned int
      mnBAGlobalForKF;  // LoopClosing 运行全局 BA 时，当前回环 KF id

  static std::mutex mGlobalMutex;  // 特征点全局锁  注意，锁本身是全局变量

 protected:
  /* 位置 */
  // Position in absolute coordinates
  cv::Mat mWorldPos;  // 绝对坐标

  /* 特征与观测 */
  // Best descriptor to fast matching
  cv::Mat mDescriptor;  // 最佳描述子
                        // 与其他描述子汉明距离中位数最小的描述子作为最佳描述子
  // Keyframes observing the point and associated index in keyframe
  std::map<KeyFrame*, size_t>
      mObservations;  // 观测关键帧集   映射<关键帧指针,
                      // 此地图点在此关键帧的特征点集合中的序号>
  // Reference KeyFrame
  KeyFrame* mpRefKF;  // 参考关键帧
  // Mean viewing direction
  cv::Mat mNormalVector;  // 平均观测方向  相机指向地图点
  // Scale invariance distances
  float mfMinDistance;  // 最近尺度无关距离
  float mfMaxDistance;  // 最远尺度无关距离

  /* 观测统计，用在 Tracking 中 */
  // Tracking counters
  int mnVisible;  // 可视次数
                  // 可视代表地图点位于帧视野范围内，但未必能成功提取特征
                  // 可视通过 Frame::isInFrustum 判断
  int mnFound;  // 检测次数  检测代表地图点在某个帧内能成功提取特征，成为关键点

  /* 其他 */
  // Bad flag (we do not currently erase MapPoint from memory)
  bool mbBad;            // 坏点标志，代表当前点的信息被删除了
  MapPoint* mpReplaced;  // 替换当前点的地图点

  Map* mpMap;  // 地图

  std::mutex mMutexPos;       // 位姿锁
  std::mutex mMutexFeatures;  // 特征锁
};

}  // namespace ORB_SLAM2

#endif  // MAPPOINT_H
