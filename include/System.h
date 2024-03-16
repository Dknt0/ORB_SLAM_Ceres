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

#ifndef SYSTEM_H
#define SYSTEM_H

#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <string>
#include <thread>

#include "FrameDrawer.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapDrawer.h"
#include "ORBVocabulary.h"
#include "Tracking.h"
#include "Viewer.h"

namespace ORB_SLAM2 {

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class System {
 public:
  // Input sensor
  enum eSensor { MONOCULAR = 0, STEREO = 1, RGBD = 2 };

 public:
  System(const string &strVocFile, const string &strSettingsFile,
         const eSensor sensor, const bool bUseViewer = true);

  cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                      const double &timestamp);
  cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap,
                    const double &timestamp);
  cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);

  void ActivateLocalizationMode();
  void DeactivateLocalizationMode();

  bool MapChanged();
  void Reset();
  void Shutdown();

  void SaveTrajectoryTUM(const string &filename);
  void SaveKeyFrameTrajectoryTUM(const string &filename);
  void SaveTrajectoryKITTI(const string &filename);

  // TODO: Save/Load functions
  // SaveMap(const string &filename);
  // LoadMap(const string &filename);

  int GetTrackingState();

  std::vector<MapPoint *> GetTrackedMapPoints();
  std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

 private:
  /* 数据库 */
  eSensor mSensor;  // 传感器类型  MONOCULAR=0, STEREO=1, RGBD=2
  ORBVocabulary *mpVocabulary;           // 视觉字典
  KeyFrameDatabase *mpKeyFrameDatabase;  // 关键帧数据库
  Map *mpMap;                            // 地图

  /* 线程类与工具类 */
  Tracking *mpTracker;          // 追踪器
  LocalMapping *mpLocalMapper;  // 局部建图器
  LoopClosing *mpLoopCloser;    // 回环检测器
  Viewer *mpViewer;             // 绘图器
  FrameDrawer *mpFrameDrawer;   // 帧绘制器  用于绘制 KF
  MapDrawer *mpMapDrawer;       // 地图绘制器  用于绘制 MP

  /* 线程 */
  std::thread *mptLocalMapping;  // LocalMapping 线程
  std::thread *mptLoopClosing;   //  LoopClosing 线程
  std::thread *mptViewer;        // 显示线程

  std::mutex mMutexReset;  // 重置互斥锁
  bool mbReset;            // 重置系统标志

  std::mutex mMutexMode;              // 模式互斥锁
  bool mbActivateLocalizationMode;    // 激活纯定位模式标志
  bool mbDeactivateLocalizationMode;  // 关闭纯定位模式标志

  // 这些状态由 Tracker 更新而来
  int mTrackingState;                             // 追踪状态
  std::vector<MapPoint *> mTrackedMapPoints;      // 当前帧观测到的 MP
  std::vector<cv::KeyPoint> mTrackedKeyPointsUn;  // 当前帧去畸变 KP
  std::mutex mMutexState;                         // 状态互斥锁
};

}  // namespace ORB_SLAM2

#endif  // SYSTEM_H
