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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include <mutex>
#include <thread>

#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"
// #include "CeresOptimizer.h"
#include "sim3.h"


class Sim3;
namespace ORB_SLAM2 {

class Tracking;
class LocalMapping;
class KeyFrameDatabase;

// typedef g2o::Sim3 Sim3;

class LoopClosing {
 public:
  typedef pair<set<KeyFrame*>, int> ConsistentGroup;  // 共视组  pair<共视 KF 集合, 累积一致性> 共视组累积一致性指包含相同 KF 的共视组的数量

  // 下述内存申请器用于为 Eigen 矩阵申请对齐的内存，加速运算

  typedef map<KeyFrame*, Sim3, std::less<KeyFrame*>, Eigen::aligned_allocator<std::pair<KeyFrame* const, Sim3> > > KeyFrameAndPose;  // 位姿映射  map<KF, Sim3, 比较器, 内存申请器>

 public:
  LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,
              const bool bFixScale);

  void Run();
  void InsertKeyFrame(KeyFrame* pKF);

  void RunGlobalBundleAdjustment(unsigned long nLoopKF);

  void SetTracker(Tracking* pTracker);
  void SetLocalMapper(LocalMapping* pLocalMapper);

  void RequestFinish();
  void RequestReset();

  bool isFinished();

  /// @brief 是否正在运行全局 BA
  /// @return
  bool isRunningGBA() {
    unique_lock<std::mutex> lock(mMutexGBA);
    return mbRunningGBA;
  }

  /// @brief 全局 BA 是否结束
  /// @return
  bool isFinishedGBA() {
    unique_lock<std::mutex> lock(mMutexGBA);
    return mbFinishedGBA;
  }

  // Eigen 内存对齐宏
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  bool DetectLoop();
  void CorrectLoop();
  void SearchAndFuse(const KeyFrameAndPose& CorrectedPosesMap);

  bool ComputeSim3();

  bool CheckNewKeyFrames();
  void ResetIfRequested();
  bool CheckFinish();
  void SetFinish();

  /* ORB 中其他模块 */
  Map* mpMap;                      // 地图
  Tracking* mpTracker;             // 追踪器
  LocalMapping* mpLocalMapper;     // 局部建图器
  KeyFrameDatabase* mpKeyFrameDB;  // 关键帧数据库
  ORBVocabulary* mpORBVocabulary;  // 视觉字典

  /* 待检测 KF 相关 */
  std::list<KeyFrame*> mlpLoopKeyFrameQueue;  // 回环 KF 队列
  KeyFrame* mpCurrentKF;                      // 当前 KF
  KeyFrame* mpMatchedKF;                      // 匹配 KF
  std::mutex mMutexLoopQueue;                 // 回环队列互斥锁

  /* 共视组相关 */
  float mnCovisibilityConsistencyTh;  // 共视一致性阈值  =3
  std::vector<ConsistentGroup> mvConsistentGroups;  // 历史共视组集  仅在回环检测中使用  vector<pair<共视 KF 集合, 与其他组的共视数量>>  数据存储策略？
  std::vector<KeyFrame*> mvpEnoughConsistentCandidates;  // 满足一致性阈值的候选 KF 集

  /* 当前 KF 回环相关 */
  std::vector<KeyFrame*> mvpCurrentConnectedKFs;  // 当前 KF 局部范围内 KF  包括当前 KF 及其近邻 KF
  std::vector<MapPoint*> mvpCurrentMatchedPoints;  // 当前 KF 对回环匹配 KF MP 匹配
  std::vector<MapPoint*> mvpLoopMapPoints;  // 回环局部 MP  匹配 KF 局部范围内 MP
  cv::Mat mScw;  // 当前 KF 位姿 Scw
  Sim3 mg2oScw;  // 当前 KF 位姿 g2o Scw

  long unsigned int mLastLoopKFid;  // 上一次闭环 KF idx

  /* 标志位 */
  bool mbResetRequested;   // 重置请求标志位
  std::mutex mMutexReset;  // 重置互斥锁

  bool mbFinishRequested;   // 终止请求标志位
  bool mbFinished;          // 终止标志位
  std::mutex mMutexFinish;  // 终止互斥锁

  bool mbRunningGBA;         // 全局 BA 运行标志位
  bool mbFinishedGBA;        // 全局 BA 结束标志位
  bool mbStopGBA;            // 暂停全局 BA  控制 g2o 中断
  std::thread* mpThreadGBA;  // 全局 BA 线程
  std::mutex mMutexGBA;      // 全局 BA 的互斥锁

  bool mbFixScale;  // 固定尺度标志位
  // 源码中 mnFullBAIdx 是 bool，应该是写错了
  int mnFullBAIdx;  // 全局 BA 优化次数
};

}  // namespace ORB_SLAM2

#endif  // LOOPCLOSING_H
