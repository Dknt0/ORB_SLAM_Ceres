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

#ifndef VIEWER_H
#define VIEWER_H

#include <mutex>

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "System.h"
#include "Tracking.h"

namespace ORB_SLAM2 {

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;

/// @brief 显示类
class Viewer {
 public:
  Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer,
         Tracking* pTracking, const string& strSettingPath);

  // Main thread function. Draw points, keyframes, the current camera pose and
  // the last processed frame. Drawing is refreshed according to the camera fps.
  // We use Pangolin.

  void Run();

  void RequestFinish();

  void RequestStop();

  bool isFinished();

  bool isStopped();

  void Release();

 private:
  bool Stop();

  System* mpSystem;
  FrameDrawer* mpFrameDrawer;
  MapDrawer* mpMapDrawer;
  Tracking* mpTracker;

  // 1/fps in ms
  double mT;  // 相机周期
  float mImageWidth;
  float mImageHeight;

  float mViewpointX;
  float mViewpointY;
  float mViewpointZ;
  float mViewpointF;

  bool CheckFinish();
  void SetFinish();

  /* 终止相关 */
  bool mbFinishRequested;  // 终止请求标志位
  bool mbFinished;  // 终止标志位
  std::mutex mMutexFinish;  // 终止互斥锁

  /* 暂停相关 */
  bool mbStopped;  // 暂停标志位
  bool mbStopRequested;  // 暂停请求标志位
  std::mutex mMutexStop;  // 暂停互斥锁
};

}  // namespace ORB_SLAM2

#endif  // VIEWER_H
