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

#ifndef TRACKING_H
#define TRACKING_H

#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Frame.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapDrawer.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "System.h"
#include "Viewer.h"
// #include "CeresOptimizer.h"

namespace ORB_SLAM2 {

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking {
 public:
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer,
           MapDrawer* pMapDrawer, Map* pMap, KeyFrameDatabase* pKFDB,
           const string& strSettingPath, const int sensor);

  /* 追踪函数 */

  cv::Mat GrabImageStereo(const cv::Mat& imRectLeft, const cv::Mat& imRectRight,
                          double timestamp);
  cv::Mat GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                        double timestamp);
  cv::Mat GrabImageMonocular(const cv::Mat& im, double timestamp);

  /* 指针设置 */

  void SetLocalMapper(LocalMapping* pLocalMapper);
  void SetLoopClosing(LoopClosing* pLoopClosing);
  void SetViewer(Viewer* pViewer);

  // The focal lenght should be similar or scale prediction will fail when
  // projecting points
  // @todo: Modify MapPoint::PredictScale to take into account focal lenght

  void ChangeCalibration(const string& strSettingPath);
  void InformOnlyTracking(const bool& flag);
  void Reset();

 public:
  // 跟踪状态
  enum eTrackingState {
    SYSTEM_NOT_READY = -1,  // 系统未就绪  貌似没用到
    NO_IMAGES_YET = 0,      // 未收到图片
    NOT_INITIALIZED = 1,    // 未初始化
    OK = 2,                 // 正常
    LOST = 3                // 追踪失败
  };

  eTrackingState mState;               // 当前追踪状态
  eTrackingState mLastProcessedState;  // 上一帧追踪状态

  int mSensor;  // @brief 传感器类型，MONOCULAR=0, STEREO=1, RGBD=2

  Frame mCurrentFrame;  // 当前帧
  cv::Mat mImGray;      // 左图

  /* 单目初始化成员变量 */
  Frame mInitialFrame;  // 初始化参考帧
  std::vector<int>
      mvIniMatches;  // 初始化匹配  参考 F KP 匹配到当前 F KP 按参考 F KP 索引
  std::vector<int> mvIniLastMatches;  // 当前 F KP 匹配到上一 F KP 按当前 F KP
                                      // 索引，记录上一 F KP 序号   貌似没有用到
  std::vector<cv::Point2f>
      mvbPrevMatched;  // 参考 F KP 在当前 F 中预匹配坐标  按参考 F KP 索引
  std::vector<cv::Point3f> mvIniP3D;  // 三角化 KP 坐标 tw

  /* 帧位姿信息 用于记录轨迹 */
  // Lists used to recover the full camera trajectory at the end of the
  // execution. Basically we store the reference keyframe for each frame and its
  // relative transformation
  list<cv::Mat> mlRelativeFramePoses;  // F 相对于 KF 位姿 TFfF  按 F 顺序索引
  list<KeyFrame*> mlpReferences;  // 参考 KF  按 F 顺序索引
  list<double> mlFrameTimes;      // 帧时间戳
  list<bool> mlbLost;             // 帧追踪状态

  bool mbOnlyTracking;  // 纯定位模式标志位

 protected:
  void Track();

  bool TrackReferenceKeyFrame();
  bool TrackWithMotionModel();
  bool Relocalization();
  bool TrackLocalMap();

  void StereoInitialization();
  void MonocularInitialization();
  void CreateInitialMapMonocular();

  bool NeedNewKeyFrame();
  void CreateNewKeyFrame();

  void CheckReplacedInLastFrame();

  void UpdateLastFrame();

  void UpdateLocalMap();
  void UpdateLocalKeyFrames();
  void UpdateLocalPoints();
  void SearchLocalPoints();

  void FreeInitializer();

  // In case of performing only localization, this flag is true when there are
  // no matches to points in the map. Still tracking will continue if there are
  // enough matches with temporal points. In that case we are doing visual
  // odometry. The system will try to do relocalization to recover "zero-drift"
  // localization to the map.
  bool mbVO;  // 纯定位模式追踪到 MP 数量不足标志位  这个状态为 false
              // 是正常的定位模式

  /* ORB 中其他对象的指针 */
  LocalMapping* mpLocalMapper = NULL;  // 局部建图
  LoopClosing* mpLoopClosing = NULL;   // 回环检测
  System* mpSystem = NULL;             // 系统
  Viewer* mpViewer = NULL;             // 显示器
  FrameDrawer* mpFrameDrawer = NULL;   // 帧绘制器
  MapDrawer* mpMapDrawer = NULL;       // 地图绘制器
  Initializer* mpInitializer = NULL;   // 单目初始化器

  /* ORB 特征提取器 */
  ORBextractor* mpORBextractorLeft = NULL;   // 左图提取器
  ORBextractor* mpORBextractorRight = NULL;  // 右图提取器
  ORBextractor* mpIniORBextractor = NULL;    // 单目初始化提取器

  /* 数据库 */
  ORBVocabulary* mpORBVocabulary = NULL;  // 视觉字典
  KeyFrameDatabase* mpKeyFrameDB = NULL;  // 关键帧数据库
  Map* mpMap = NULL;                      // 地图

  /* 配置参数 */
  cv::Mat mK;             // 相机内参
  cv::Mat mDistCoef;      // 相机畸变系数
  float mbf;              // 双目基线 * fx
  int mMinFrames;         // 关键帧最小间隔帧数  =0
  int mMaxFrames;         // 关键帧最大间隔帧数  =fps
  float mThDepth;         // 远双目点阈值  米
  float mDepthMapFactor;  // 深度因子  1m 对应的灰度值
  bool mbRGB;             // 彩色图通道顺序  true-RGB, false-BGR

  /* 追踪过程变量 */
  int mnMatchesInliers;  // 当前帧匹配 KP 数量  局部地图匹配与 KF 判断中使用
  KeyFrame* mpLastKeyFrame = NULL;  // 上一 KF
  Frame mLastFrame;                 // 上一 F
  unsigned int mnLastKeyFrameId;    // 上一个 KF id
  unsigned int mnLastRelocFrameId;  // 上一次重定位 F id
  cv::Mat mVelocity;  // 两帧间的相对位姿 上一帧相对当前帧  Tcccl

  KeyFrame* mpReferenceKF = NULL;     // 当前参考 KF
  list<MapPoint*> mlpTemporalPoints;  // 临时 MP  用于 VIO 帧间匹配
  // 局部地图
  std::vector<KeyFrame*> mvpLocalKeyFrames;  // 局部地图 KF
  std::vector<MapPoint*> mvpLocalMapPoints;  // 局部地图 MP
};

}  // namespace ORB_SLAM2

#endif  // TRACKING_H
