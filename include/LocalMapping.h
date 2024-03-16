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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include <mutex>

#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "LoopClosing.h"
#include "Map.h"
#include "Tracking.h"

namespace ORB_SLAM2 {

class Tracking;
class LoopClosing;
class Map;

class LocalMapping {
 public:
  LocalMapping(Map* pMap, const float bMonocular);

  void Run();
  void InsertKeyFrame(KeyFrame* pKF);

  void SetLoopCloser(LoopClosing* pLoopCloser);
  void SetTracker(Tracking* pTracker);

  /// 请求接口

  void RequestStop();
  void RequestReset();
  void RequestFinish();
  bool SetNotStop(bool flag);
  void SetAcceptKeyFrames(bool flag);
  void InterruptBA();
  void Release();

  /// 查询接口

  bool Stop();
  bool isStopped();
  bool stopRequested();
  bool AcceptKeyFrames();
  bool isFinished();

  /// @brief 查询 KF 队列长度
  /// @return
  int KeyframesInQueue() {
    unique_lock<std::mutex> lock(mMutexNewKFs);
    return mlNewKeyFrames.size();
  }

 protected:
  /// 内部功能函数

  bool CheckNewKeyFrames();
  void ProcessNewKeyFrame();
  void CreateNewMapPoints();
  void MapPointCulling();
  void SearchInNeighbors();
  void KeyFrameCulling();

  /// 数学工具

  cv::Mat ComputeF12(KeyFrame*& pKF1, KeyFrame*& pKF2);
  cv::Mat SkewSymmetricMatrix(const cv::Mat& v);

  void ResetIfRequested();
  bool CheckFinish();
  void SetFinish();

  /* ORB 中其他对象指针 */
  bool mbMonocular;           // 单目标志位
  Map* mpMap;                 // 地图
  Tracking* mpTracker;        // 追踪器
  LoopClosing* mpLoopCloser;  // 回环检测器

  /* KF 与 MP */
  std::list<KeyFrame*> mlNewKeyFrames;           // 新 KF 队列
  KeyFrame* mpCurrentKeyFrame;                   // 当前 KF
  std::list<MapPoint*> mlpRecentAddedMapPoints;  // 新增 MP 集
  std::mutex mMutexNewKFs;                       // 新增 KF 互斥锁

  /* 重置相关 */
  bool mbResetRequested;   // 重置请求标志位
  std::mutex mMutexReset;  // 重置互斥锁

  /* 终止相关 */
  bool mbFinishRequested;   // 终止请求标志位
  bool mbFinished;          // 终止标志位
  std::mutex mMutexFinish;  // 终止互斥锁

  /* 暂停相关 */
  bool mbStopped;         // 暂停标志位
  bool mbStopRequested;   // 暂停请求标志位
  bool mbNotStop;         // 拒绝暂停标志位
  std::mutex mMutexStop;  // 暂停互斥锁

  /* KF 接收相关 */
  bool mbAbortBA;           // 中断 BA 标志位  控制 g2o 中断
  bool mbAcceptKeyFrames;   // 可以接受 KF 标志位
  std::mutex mMutexAccept;  // 接收 KF 互斥锁
};

}  // namespace ORB_SLAM2

#endif  // LOCALMAPPING_H
