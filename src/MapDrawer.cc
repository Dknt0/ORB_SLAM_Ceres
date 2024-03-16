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

#include "MapDrawer.h"

#include <pangolin/pangolin.h>

#include <mutex>

#include "KeyFrame.h"
#include "MapPoint.h"

namespace ORB_SLAM2 {

/// @brief 地图绘制器构造函数
/// @param pMap
/// @param strSettingPath
MapDrawer::MapDrawer(Map *pMap, const string &strSettingPath) : mpMap(pMap) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
  mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
  mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
  mPointSize = fSettings["Viewer.PointSize"];
  mCameraSize = fSettings["Viewer.CameraSize"];
  mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
}

/// @brief 绘制地图点
void MapDrawer::DrawMapPoints() {
  const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();  // 地图点
  const vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints();  // 局部地图点
  
  set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

  if (vpMPs.empty()) return;

  /* 绘制一般地图点 */
  glPointSize(mPointSize);
  glBegin(GL_POINTS);
  // 白色
  glColor3f(1.0, 1.0, 1.0);

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i])) continue;
    cv::Mat pos = vpMPs[i]->GetWorldPos();
    glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
  }
  glEnd();

  /* 绘制局部地图点 */
  glPointSize(mPointSize);
  glBegin(GL_POINTS);
  glColor3f(1.0, 0.0, 0.0);  // 红色

  for (set<MapPoint *>::iterator sit = spRefMPs.begin(), send = spRefMPs.end();
       sit != send; sit++) {
    if ((*sit)->isBad()) continue;
    cv::Mat pos = (*sit)->GetWorldPos();
    glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
  }

  glEnd();
}

/// @brief 绘制 KF 和共视图
/// @param bDrawKF 绘制 KF 标志位
/// @param bDrawGraph 绘制共视图标志位
void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph) {
  const float &w = mKeyFrameSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  const vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();

  // 绘制 KF
  if (bDrawKF) {
    for (size_t i = 0; i < vpKFs.size(); i++) {
      KeyFrame *pKF = vpKFs[i];
      // OpenGL Column-Major 所以要转置
      cv::Mat Twc = pKF->GetPoseInverse().t();

      // OpenGL 矩阵乘法貌似是用栈实现的，相比直接计算可能会有加速？
      glPushMatrix();  // 矩阵入栈

      glMultMatrixf(Twc.ptr<GLfloat>(0));  // 左乘位姿矩阵

      // 画框  四棱锥
      glLineWidth(mKeyFrameLineWidth);
      glColor3f(0.0f, 0.0f, 1.0f);
      glBegin(GL_LINES);
      glVertex3f(0, 0, 0);
      glVertex3f(w, h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, h, z);

      glVertex3f(w, h, z);
      glVertex3f(w, -h, z);

      glVertex3f(-w, h, z);
      glVertex3f(-w, -h, z);

      glVertex3f(-w, h, z);
      glVertex3f(w, h, z);

      glVertex3f(-w, -h, z);
      glVertex3f(w, -h, z);
      glEnd();

      glPopMatrix();  // 矩阵出栈
    }
  }

  // 绘制共视图
  if (bDrawGraph) {
    // glLineWidth(mGraphLineWidth);
    // glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
    glBegin(GL_LINES);

    for (size_t i = 0; i < vpKFs.size(); i++) {
      // Covisibility Graph
      glLineWidth(mGraphLineWidth);
      glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
      const vector<KeyFrame *> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);  // 本质图边  100 阈值是本质图
      cv::Mat Ow = vpKFs[i]->GetCameraCenter();
      if (!vCovKFs.empty()) {
        for (vector<KeyFrame *>::const_iterator vit = vCovKFs.begin(),
                                                vend = vCovKFs.end();
             vit != vend; vit++) {
          if ((*vit)->mnId < vpKFs[i]->mnId) continue;  // 只画一次
          cv::Mat Ow2 = (*vit)->GetCameraCenter();
          glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
          glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
        }
      }

      // Spanning tree
      glLineWidth(mGraphLineWidth * 3);
      glColor4f(1.0f, 1.0f, 0.0f, 0.6f);
      KeyFrame *pParent = vpKFs[i]->GetParent();
      if (pParent) {
        cv::Mat Owp = pParent->GetCameraCenter();
        glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
        glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
      }

      // Loops
      glLineWidth(mGraphLineWidth * 3);
      glColor4f(1.0f, 0.0f, 1.0f, 0.6f);
      set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
      for (set<KeyFrame *>::iterator sit = sLoopKFs.begin(),
                                     send = sLoopKFs.end();
           sit != send; sit++) {
        if ((*sit)->mnId < vpKFs[i]->mnId) continue;
        cv::Mat Owl = (*sit)->GetCameraCenter();
        glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
        glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
      }
    }

    glEnd();
  }
}

/// @brief 绘制当前相机
/// @param Twc 
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc) {
  const float &w = mCameraSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  glPushMatrix();

#ifdef HAVE_GLES
  glMultMatrixf(Twc.m);
#else
  glMultMatrixd(Twc.m);
#endif

  glLineWidth(mCameraLineWidth);
  glColor3f(0.0f, 1.0f, 0.0f);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);

  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);

  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();

  glPopMatrix();
}

/// @brief 设置当前相机位姿
/// @param Tcw 
void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw) {
  unique_lock<mutex> lock(mMutexCamera);
  mCameraPose = Tcw.clone();
}

/// @brief 获取当前相机位姿的 OpenGl 矩阵
/// @param M 相机位姿  输出
void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M) {
  if (!mCameraPose.empty()) {
    cv::Mat Rwc(3, 3, CV_32F);
    cv::Mat twc(3, 1, CV_32F);
    {
      unique_lock<mutex> lock(mMutexCamera);
      Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
      twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
    }

    // Column-Major
    M.m[0] = Rwc.at<float>(0, 0);
    M.m[1] = Rwc.at<float>(1, 0);
    M.m[2] = Rwc.at<float>(2, 0);
    M.m[3] = 0.0;

    M.m[4] = Rwc.at<float>(0, 1);
    M.m[5] = Rwc.at<float>(1, 1);
    M.m[6] = Rwc.at<float>(2, 1);
    M.m[7] = 0.0;

    M.m[8] = Rwc.at<float>(0, 2);
    M.m[9] = Rwc.at<float>(1, 2);
    M.m[10] = Rwc.at<float>(2, 2);
    M.m[11] = 0.0;

    M.m[12] = twc.at<float>(0);
    M.m[13] = twc.at<float>(1);
    M.m[14] = twc.at<float>(2);
    M.m[15] = 1.0;
  } else
    M.SetIdentity();
}

}  // namespace ORB_SLAM2
