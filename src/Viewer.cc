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

#include "Viewer.h"

#include <pangolin/pangolin.h>

#include <mutex>

namespace ORB_SLAM2 {

/// @brief 显示类构造函数
/// @param pSystem
/// @param pFrameDrawer
/// @param pMapDrawer
/// @param pTracking
/// @param strSettingPath
Viewer::Viewer(System *pSystem, FrameDrawer *pFrameDrawer,
               MapDrawer *pMapDrawer, Tracking *pTracking,
               const string &strSettingPath)
    : mpSystem(pSystem),
      mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer),
      mpTracker(pTracking),
      mbFinishRequested(false),
      mbFinished(true),
      mbStopped(true),
      mbStopRequested(false) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  float fps = fSettings["Camera.fps"];
  if (fps < 1) fps = 30;
  mT = 1e3 / fps;

  mImageWidth = fSettings["Camera.width"];
  mImageHeight = fSettings["Camera.height"];
  if (mImageWidth < 1 || mImageHeight < 1) {
    mImageWidth = 640;
    mImageHeight = 480;
  }

  mViewpointX = fSettings["Viewer.ViewpointX"];
  mViewpointY = fSettings["Viewer.ViewpointY"];
  mViewpointZ = fSettings["Viewer.ViewpointZ"];
  mViewpointF = fSettings["Viewer.ViewpointF"];
}

/// @brief 运行  线程函数
void Viewer::Run() {
  mbFinished = false;
  mbStopped = false;

  // 创建窗口
  pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);

  // 设置 OpenGL 属性
  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  // Issue specific OpenGl we might need
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // 创建面板，定义 UI
  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true,
                                       true);  // 跟随相机 UI 标志位
  pangolin::Var<bool> menuShowPoints("menu.Show Points", true,
                                     true);  // 显示 MP UI 标志位
  pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true,
                                        true);  // 显示 KF UI 标志位
  pangolin::Var<bool> menuShowGraph("menu.Show Graph", true,
                                    true);  // 显示共视图 UI 标志位
  pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false,
                                           true);  // 纯定位模式 UI 标志位
  pangolin::Var<bool> menuReset("menu.Reset", false, false);  // 重置 UI 标志位

  // 定义相机渲染对象
  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389,
                                 0.1, 3000),
      pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0,
                                0.0, -1.0, 0.0));

  // 创建交互视图
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175),
                                         1.0, -1024.0f / 768.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();

  cv::namedWindow("ORB-SLAM2: Current Frame");

  bool bFollow = true;             // 跟随标志位
  bool bLocalizationMode = false;  // 定位标志位

  while (1) {
    /* Pangolin 显示 */
    // 每一帧都是重复绘制的
    // 清空颜色、深度缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 获取当前相机位姿
    mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

    // 相机跟随设置
    if (menuFollowCamera && bFollow) {
      // 面板为跟随，标志位为跟随
      s_cam.Follow(Twc);
    } else if (menuFollowCamera && !bFollow) {
      // 开启跟随
      s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
          mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
      s_cam.Follow(Twc);
      bFollow = true;
    } else if (!menuFollowCamera && bFollow) {
      // 关闭跟随
      bFollow = false;
    }

    // 纯定位模式设置
    if (menuLocalizationMode && !bLocalizationMode) {
      // 开启纯定位模式
      mpSystem->ActivateLocalizationMode();
      bLocalizationMode = true;
    } else if (!menuLocalizationMode && bLocalizationMode) {
      // 关闭纯定位模式
      mpSystem->DeactivateLocalizationMode();
      bLocalizationMode = false;
    }

    d_cam.Activate(s_cam);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    mpMapDrawer->DrawCurrentCamera(Twc);  // 绘制相机
    if (menuShowKeyFrames || menuShowGraph)
      mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);  // 绘制 KF
    if (menuShowPoints) mpMapDrawer->DrawMapPoints();  // 绘制 MP

    pangolin::FinishFrame();

    /* OpenCV 显示 */
    cv::Mat im = mpFrameDrawer->DrawFrame();
    cv::imshow("ORB-SLAM2: Current Frame", im);
    cv::waitKey(mT);

    // 检查是否重置
    if (menuReset) {
      menuShowGraph = true;
      menuShowKeyFrames = true;
      menuShowPoints = true;
      menuLocalizationMode = false;
      if (bLocalizationMode) mpSystem->DeactivateLocalizationMode();
      bLocalizationMode = false;
      bFollow = true;
      menuFollowCamera = true;
      mpSystem->Reset();
      menuReset = false;
    }

    if (Stop()) {
      while (isStopped()) {
        usleep(3000);
      }
    }

    if (CheckFinish()) break;
  }

  SetFinish();
}

/// @brief 请求终止
void Viewer::RequestFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

/// @brief 检查终止
/// @return
bool Viewer::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

/// @brief 设置终止
void Viewer::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
}

/// @brief 是否已终止
/// @return
bool Viewer::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

/// @brief 请求暂停
void Viewer::RequestStop() {
  unique_lock<mutex> lock(mMutexStop);
  if (!mbStopped) mbStopRequested = true;
}

/// @brief 是否已暂停
/// @return
bool Viewer::isStopped() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopped;
}

/// @brief 按请求暂停
/// @return
bool Viewer::Stop() {
  unique_lock<mutex> lock(mMutexStop);
  unique_lock<mutex> lock2(mMutexFinish);

  if (mbFinishRequested)
    return false;
  else if (mbStopRequested) {
    mbStopped = true;
    mbStopRequested = false;
    return true;
  }

  return false;
}

/// @brief 释放
void Viewer::Release() {
  unique_lock<mutex> lock(mMutexStop);
  mbStopped = false;
}

}  // namespace ORB_SLAM2
