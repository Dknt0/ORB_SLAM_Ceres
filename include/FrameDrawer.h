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

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include <opencv2/imgproc/types_c.h>

#include <mutex>
#include <opencv2/opencv.hpp>

#include "Map.h"
#include "MapPoint.h"
#include "Tracking.h"

namespace ORB_SLAM2 {

class Tracking;
class Viewer;

/// FrameDrawer 是在 OpenCV::Mat 上的操作

/// @brief 帧绘制器
class FrameDrawer {
 public:
  FrameDrawer(Map *pMap);

  // Update info from the last processed frame.
  void Update(Tracking *pTracker);

  // Draw last processed frame.
  cv::Mat DrawFrame();

 protected:
  void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

  // Info of the frame to be drawn
  cv::Mat mIm;  // 图片
  int N;  // KP 数量
  vector<cv::KeyPoint> mvCurrentKeys;  // 当前 KP
  vector<bool> mvbMap;  // KP 匹配到 MP 标志位
  vector<bool> mvbVO;  // KP 匹配到临时 MP 标志位
  bool mbOnlyTracking;  // 纯定位模式
  int mnTracked;  // 匹配到 MP KP 总数
  int mnTrackedVO;  // 匹配到临时 MP KP 总数
  vector<cv::KeyPoint> mvIniKeys;  // 初始 KP
  vector<int> mvIniMatches;  // 初始匹配
  int mState;  // Tracking State

  Map *mpMap;

  std::mutex mMutex;  // 互斥锁
};

}  // namespace ORB_SLAM2

#endif  // FRAMEDRAWER_H
