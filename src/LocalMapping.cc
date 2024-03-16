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

#include "LocalMapping.h"

#include <mutex>

#include "CeresOptimizer.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2 {

/// @brief 构造函数
/// @param pMap 地图
/// @param bMonocular 单目标志位
LocalMapping::LocalMapping(Map *pMap, const float bMonocular)
    : mbMonocular(bMonocular),
      mpMap(pMap),
      mbResetRequested(false),
      mbFinishRequested(false),
      mbFinished(true),
      mbStopped(false),
      mbStopRequested(false),
      mbNotStop(false),
      mbAbortBA(false),
      mbAcceptKeyFrames(true) {}

/// @brief 设置 LoopClosing 线程
/// @param pLoopCloser
void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser) {
  mpLoopCloser = pLoopCloser;
}

/// @brief 设置 Tracking 线程
/// @param pTracker
void LocalMapping::SetTracker(Tracking *pTracker) { mpTracker = pTracker; }

/// @brief 运行  线程函数
void LocalMapping::Run() {
  mbFinished = false;

  while (1) {
    // 拒绝接受 KF
    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(false);

    // 如果队列中存在新 KF
    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      // 处理新 KF
      // BoW conversion and insertion in Map
      ProcessNewKeyFrame();

      // 剔除 MP
      // Check recent MapPoints
      MapPointCulling();

      // 创建新 MP
      // Triangulate new MapPoints
      CreateNewMapPoints();

      // 如果 KF 队列空
      if (!CheckNewKeyFrames()) {
        // 依据共视关系搜索、融合 MP
        // Find more matches in neighbor keyframes and fuse point duplications
        SearchInNeighbors();
      }

      mbAbortBA = false;

      // 如果 KF 队列空 且 没有暂停请求
      if (!CheckNewKeyFrames() && !stopRequested()) {
        // 进行局部 BA
        // Local BA
        if (mpMap->KeyFramesInMap() > 2) {
          // Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA,
          //                                  mpMap);
                                           
          CeresOptimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA,
                                           mpMap);
        }

        // 关键帧剔除
        // Check redundant local Keyframes
        KeyFrameCulling();
      }

      // 将关键帧传入闭环检测器
      mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
    }
    // 如果需要暂停
    else if (Stop()) {
      // 暂停循环
      // Safe area to stop
      while (isStopped() && !CheckFinish()) {
        usleep(3000);
      }
      if (CheckFinish()) break;
    }

    // 重置检查
    ResetIfRequested();

    // 恢复接受 KF
    // Tracking will see that Local Mapping is not busy
    SetAcceptKeyFrames(true);

    // 终止检查
    if (CheckFinish()) break;

    // 延时   Tracking 需要在这 3000 us 中插入新关键帧
    usleep(3000);
  }

  // 线程结束
  SetFinish();
}

/// @brief 添加 KF 到队列 中断 BA
/// @param pKF 新 KF
void LocalMapping::InsertKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexNewKFs);
  mlNewKeyFrames.push_back(pKF);
  mbAbortBA = true;
}

/// @brief KF 队列中是否有 KF
/// @return
bool LocalMapping::CheckNewKeyFrames() {
  unique_lock<mutex> lock(mMutexNewKFs);
  return (!mlNewKeyFrames.empty());
}

/// @brief 处理新 KF
void LocalMapping::ProcessNewKeyFrame() {
  // 获取队首 KF
  {
    unique_lock<mutex> lock(mMutexNewKFs);
    mpCurrentKeyFrame = mlNewKeyFrames.front();
    mlNewKeyFrames.pop_front();
  }

  // Compute Bags of Words structures
  mpCurrentKeyFrame->ComputeBoW();

  // 关联 MP 与当前 KF，更新深度、法线、描述子
  // Associate MapPoints to the new keyframe and update normal and descriptor
  const vector<MapPoint *> vpMapPointMatches =
      mpCurrentKeyFrame->GetMapPointMatches();  // 当前 KF MP

  // 遍历当前 KF MP
  for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
    MapPoint *pMP = vpMapPointMatches[i];
    // 如果 MP 有效且不为坏点
    if (pMP) {
      if (!pMP->isBad()) {
        if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
          // 当前 KF 观测到 MP，但 MP 没有记录当前 KF 信息，添加之
          pMP->AddObservation(mpCurrentKeyFrame, i);
          pMP->UpdateNormalAndDepth();
          pMP->ComputeDistinctiveDescriptors();
        } else  // this can only happen for new stereo points inserted by the
                // Tracking
        {
          // 对于近双目点，添加到新增 MP 集，等待进一步处理
          mlpRecentAddedMapPoints.push_back(pMP);
        }
      }
    }
  }

  // 更新共视图关系
  // Update links in the Covisibility Graph
  mpCurrentKeyFrame->UpdateConnections();

  // 添加 KF 到地图
  // Insert Keyframe in Map
  mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/// @brief 局部范围内剔除 MP
void LocalMapping::MapPointCulling() {
  // Check Recent Added MapPoints
  list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
  const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

  int nThObs;  // MP 观测阈值
  if (mbMonocular)
    nThObs = 2;
  else
    nThObs = 3;
  const int cnThObs = nThObs;

  // 遍历新增 MP 集
  while (lit != mlpRecentAddedMapPoints.end()) {
    MapPoint *pMP = *lit;
    if (pMP->isBad()) {
      // 坏点
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (pMP->GetFoundRatio() < 0.25f) {
      // 检测比小于 0.25 的地图点，设置为坏点
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 &&
               pMP->Observations() <= cnThObs) {
      // 连续两帧没有被观测到，且 KF 观测次数小于阈值，设置为坏点
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
      // 连续三帧没有被观测到，从新增 MP 集中删除，但在地图中保留
      lit = mlpRecentAddedMapPoints.erase(lit);
    else
      lit++;
  }
}

/// @brief 三角化创建新 MP
void LocalMapping::CreateNewMapPoints() {
  /* 从共视图获取近邻 KF */
  // Retrieve neighbor keyframes in covisibility graph
  int nn = 10;  // 近邻 KF 数  单目 20 其他 10
  if (mbMonocular) nn = 20;
  const vector<KeyFrame *> vpNeighKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);  // 近邻 KF

  ORBmatcher matcher(0.6, false);

  // KF1 位姿、内参信息
  cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
  cv::Mat Rwc1 = Rcw1.t();
  cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
  cv::Mat Tcw1(3, 4, CV_32F);  // Tcw1 3*4
  Rcw1.copyTo(Tcw1.colRange(0, 3));
  tcw1.copyTo(Tcw1.col(3));
  cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

  const float &fx1 = mpCurrentKeyFrame->fx;
  const float &fy1 = mpCurrentKeyFrame->fy;
  const float &cx1 = mpCurrentKeyFrame->cx;
  const float &cy1 = mpCurrentKeyFrame->cy;
  const float &invfx1 = mpCurrentKeyFrame->invfx;
  const float &invfy1 = mpCurrentKeyFrame->invfy;

  const float ratioFactor =
      1.5f * mpCurrentKeyFrame->mfScaleFactor;  // 尺度比例阈值

  int nnew = 0;  // 新 MP 数

  // 对每一个近邻 KF 进行对极约束检查并三角化
  // Search matches with epipolar restriction and triangulate
  for (size_t i = 0; i < vpNeighKFs.size(); i++) {
    // 如果出现新 KF 返回
    if (i > 0 && CheckNewKeyFrames()) return;

    KeyFrame *pKF2 = vpNeighKFs[i];  // 近邻 KF2

    /* 检查帧间距离是否过小 */
    // Check first that baseline is not too short
    cv::Mat Ow2 = pKF2->GetCameraCenter();
    cv::Mat vBaseline = Ow2 - Ow1;
    const float baseline = cv::norm(vBaseline);

    if (!mbMonocular) {
      // 非单目情况需要大于基线长度
      if (baseline < pKF2->mb) continue;
    } else {
      const float medianDepthKF2 =
          pKF2->ComputeSceneMedianDepth(2);  // 场景深度中位数
      const float ratioBaselineDepth = baseline / medianDepthKF2;

      // 帧间距离需超过场景深度中位数的 1%
      if (ratioBaselineDepth < 0.01) continue;
    }

    /* 计算基础矩阵 */
    // Compute Fundamental Matrix
    cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

    // Search matches that fullfil epipolar constraint
    vector<pair<size_t, size_t> >
        vMatchedIndices;  // KP 匹配 v<p<KF1 idx, KF2 idx>>  结果中不包含当前 KF
                          // 中已经匹配 MP 的 KP
    /* 三角化匹配 */
    matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12,
                                   vMatchedIndices, false);

    // KF1 位姿、内参信息
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = pKF2->GetTranslation();
    cv::Mat Tcw2(3, 4, CV_32F);  // Tcw1 3*4
    Rcw2.copyTo(Tcw2.colRange(0, 3));
    tcw2.copyTo(Tcw2.col(3));

    const float &fx2 = pKF2->fx;
    const float &fy2 = pKF2->fy;
    const float &cx2 = pKF2->cx;
    const float &cy2 = pKF2->cy;
    const float &invfx2 = pKF2->invfx;
    const float &invfy2 = pKF2->invfy;

    // 三角化匹配对
    // Triangulate each match
    const int nmatches = vMatchedIndices.size();
    // 遍历匹配结果
    for (int ikp = 0; ikp < nmatches; ikp++) {
      const int &idx1 = vMatchedIndices[ikp].first;   // KF1 KP id
      const int &idx2 = vMatchedIndices[ikp].second;  // KF2 KP id

      const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];  // KF1 KP
      const float kp1_ur =
          mpCurrentKeyFrame->mvuRight[idx1];  // KF1 KP 右点坐标
      bool bStereo1 = kp1_ur >= 0;            // KF1 KP 双目点标志位

      const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];  // KF2 KP
      const float kp2_ur = pKF2->mvuRight[idx2];       // KF2 KP 右点坐标
      bool bStereo2 = kp2_ur >= 0;  // KF2 KP 双目点标志位

      // 检验光线视差
      // Check parallax between rays

      /* 计算观测方向余弦 */
      // 归一化平面坐标
      cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1,
                     (kp1.pt.y - cy1) * invfy1, 1.0);  // KF1 KP 归一化平面坐标
      cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2,
                     (kp2.pt.y - cy2) * invfy2, 1.0);  // KF2 KP 归一化平面坐标

      cv::Mat ray1 = Rwc1 * xn1;  // KF1 KP 方向向量  世界系
      cv::Mat ray2 = Rwc2 * xn2;  // KF2 KP 方向向量  世界系
      const float cosParallaxRays =
          ray1.dot(ray2) /
          (cv::norm(ray1) * cv::norm(ray2));  // KF1 KF2 KP 观测方向余弦

      /* 计算双目视差余弦 */
      // 这里的视差余弦是指以双目基线为对直角边，深度为侧直角边构成直角三角形顶角的余弦值
      float cosParallaxStereo =
          cosParallaxRays +
          1;  // 双目视差余弦，初值给一个比较大的  这个值越大，代表相对深度越大
      float cosParallaxStereo1 = cosParallaxStereo;  // KF1 双目视差余弦
      float cosParallaxStereo2 = cosParallaxStereo;  // KF2 双目视差余弦

      // 优先使用当前 KF 的双目视差余弦
      if (bStereo1)
        cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2,
                                           mpCurrentKeyFrame->mvDepth[idx1]));
      else if (bStereo2)
        cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

      // 优先使用当前 KF 的双目视差余弦，如果两个 KP
      // 都不是双目点，则双目视差余弦大于 1
      cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

      /* 深度恢复 */
      cv::Mat x3D;  // KP 空间位置
      // 三种不同深度恢复方法
      if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
          (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
        // 情形1 条件如下：
        // 1. KF1 KF2 KP 观测方向余弦小于双目视差余弦，即
        // 2. 观测方向余弦大于 0，即角度小于 90
        // 3. 其中一个 KP
        // 为双目点或观测方向余弦小于阈值，对应观测方向角度大于 1.14
        // 度，保证三角化能成功

        // 这里第三个条件中为什么考虑 KF
        // 是否为双目点？既然是三角化，那和相机类型应该是没有关系的

        // Linear Triangulation Method
        cv::Mat A(4, 4, CV_32F);
        A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
        A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
        A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
        A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        x3D = vt.row(3).t();

        if (x3D.at<float>(3) == 0) continue;

        // Euclidean coordinates
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);  // 归一化齐次坐标

      } else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
        // 情形2 KF1 深度信息有效且 KF1 双目视差余弦比 KF2 小
        // 说明 KF1 中深度更小，更准确
        x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
      } else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
        // 情形2 KF2 深度信息有效且 KF2 双目视差余弦比 KF1 小
        x3D = pKF2->UnprojectStereo(idx2);
      } else
        continue;  // No stereo and very low parallax

      cv::Mat x3Dt = x3D.t();  // KP 空间位置行向量

      /* 检验 MP 是否位于两 KF 前方 */
      // Check triangulation in front of cameras
      float z1 =
          Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);  // 相对于 KF1 z 坐标
      if (z1 <= 0) continue;

      float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
      if (z2 <= 0) continue;

      /* 检验在 KF1 中的重投影误差 */
      // Check reprojection error in first keyframe
      const float &sigmaSquare1 =
          mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];  // KP 金字塔比例平方 >1
      const float x1 =
          Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);  // KF1 归一化坐标 x
      const float y1 =
          Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);  // KF2 归一化坐标 x
      const float invz1 = 1.0 / z1;                   // KF1 逆深度

      if (!bStereo1) {
        // 单目点重投影误差计算
        float u1 = fx1 * x1 * invz1 + cx1;
        float v1 = fy1 * y1 * invz1 + cy1;
        float errX1 = u1 - kp1.pt.x;
        float errY1 = v1 - kp1.pt.y;
        if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1) continue;
      } else {
        // 双目点重投影误差计算
        float u1 = fx1 * x1 * invz1 + cx1;
        float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
        float v1 = fy1 * y1 * invz1 + cy1;
        float errX1 = u1 - kp1.pt.x;
        float errY1 = v1 - kp1.pt.y;
        float errX1_r = u1_r - kp1_ur;
        if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) >
            7.8 * sigmaSquare1)
          continue;
      }

      /* 检验在 KF2 中的重投影误差 */
      // Check reprojection error in second keyframe
      const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
      const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
      const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
      const float invz2 = 1.0 / z2;
      if (!bStereo2) {
        float u2 = fx2 * x2 * invz2 + cx2;
        float v2 = fy2 * y2 * invz2 + cy2;
        float errX2 = u2 - kp2.pt.x;
        float errY2 = v2 - kp2.pt.y;
        if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2) continue;
      } else {
        float u2 = fx2 * x2 * invz2 + cx2;
        float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
        float v2 = fy2 * y2 * invz2 + cy2;
        float errX2 = u2 - kp2.pt.x;
        float errY2 = v2 - kp2.pt.y;
        float errX2_r = u2_r - kp2_ur;
        if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) >
            7.8 * sigmaSquare2)
          continue;
      }

      /* 检验尺度一致性 */
      // Check scale consistency
      cv::Mat normal1 = x3D - Ow1;
      float dist1 = cv::norm(normal1);  // 距离

      cv::Mat normal2 = x3D - Ow2;
      float dist2 = cv::norm(normal2);

      if (dist1 == 0 || dist2 == 0) continue;

      const float ratioDist = dist2 / dist1;  // 距离比例
      const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] /
                                pKF2->mvScaleFactors[kp2.octave];  // 尺度比例

      /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
          continue;*/
      if (ratioDist * ratioFactor < ratioOctave ||
          ratioDist > ratioOctave * ratioFactor)
        continue;

      /* 三角化成功，创建 MP，添加到地图、新增 MP 集 */
      // Triangulation is succesfull
      MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

      // MP 与 KF 关联
      pMP->AddObservation(mpCurrentKeyFrame, idx1);
      pMP->AddObservation(pKF2, idx2);
      // KF 与 MP 关联
      mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
      pKF2->AddMapPoint(pMP, idx2);
      // 计算 MP 描述子、法线、深度
      pMP->ComputeDistinctiveDescriptors();
      pMP->UpdateNormalAndDepth();
      // 添加 MP 到地图、新增 MP 集
      mpMap->AddMapPoint(pMP);
      mlpRecentAddedMapPoints.push_back(pMP);

      nnew++;
    }
  }
}

/// @brief 近邻地图点匹配融合
void LocalMapping::SearchInNeighbors() {
  /* 获取目标近邻 KF 集，由当前 KF 近邻、近邻的近邻组成 */
  // Retrieve neighbor keyframes
  int nn = 10;
  if (mbMonocular) nn = 20;
  const vector<KeyFrame *> vpNeighKFs =
      mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);  // 近邻 KF 集

  vector<KeyFrame *> vpTargetKFs;  // 目标 KF 集  由当前 KF 近邻、近邻的近邻构成
  for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(),
                                          vend = vpNeighKFs.end();
       vit != vend; vit++) {
    // 将近邻 KF 添加到目标 KF 集
    KeyFrame *pKFi = *vit;
    if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
      continue;
    vpTargetKFs.push_back(pKFi);
    pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

    // 将近邻 KF 的近邻添加到目标 KF 集
    // Extend to some second neighbors
    const vector<KeyFrame *> vpSecondNeighKFs =
        pKFi->GetBestCovisibilityKeyFrames(5);  // 近邻 KF 的近邻
    for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(),
                                            vend2 = vpSecondNeighKFs.end();
         vit2 != vend2; vit2++) {
      KeyFrame *pKFi2 = *vit2;
      if (pKFi2->isBad() ||
          pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId ||
          pKFi2->mnId == mpCurrentKeyFrame->mnId)
        continue;
      vpTargetKFs.push_back(pKFi2);
    }
  }

  /* 将当前 KF MP 投影到近邻 KF 中进行融合 */
  // Search matches by projection from current KF in target KFs
  ORBmatcher matcher;
  vector<MapPoint *> vpMapPointMatches =
      mpCurrentKeyFrame->GetMapPointMatches();
  for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(),
                                    vend = vpTargetKFs.end();
       vit != vend; vit++) {
    KeyFrame *pKFi = *vit;

    matcher.Fuse(pKFi, vpMapPointMatches);
  }

  /* 整合近邻 KF MP 并投影到当前 KF 中进行融合 */
  // Search matches by projection from target KFs in current KF
  vector<MapPoint *> vpFuseCandidates;
  vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

  for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(),
                                    vendKF = vpTargetKFs.end();
       vitKF != vendKF; vitKF++) {
    KeyFrame *pKFi = *vitKF;

    vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

    for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(),
                                      vendMP = vpMapPointsKFi.end();
         vitMP != vendMP; vitMP++) {
      MapPoint *pMP = *vitMP;
      if (!pMP) continue;
      if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
        continue;
      pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
      vpFuseCandidates.push_back(pMP);
    }
  }

  matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

  /* 更新地图点观测信息，更新当前 KF 共视关系 */
  // Update points
  vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
  for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
    MapPoint *pMP = vpMapPointMatches[i];
    if (pMP) {
      if (!pMP->isBad()) {
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
      }
    }
  }

  // Update connections in covisibility graph
  mpCurrentKeyFrame->UpdateConnections();
}

/// @brief 计算 KF 间基础矩阵  K1'(^-1)*t12^R12K2'(^-1)
/// @param pKF1 KF1
/// @param pKF2 KF2
/// @return
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2) {
  cv::Mat R1w = pKF1->GetRotation();
  cv::Mat t1w = pKF1->GetTranslation();
  cv::Mat R2w = pKF2->GetRotation();
  cv::Mat t2w = pKF2->GetTranslation();

  cv::Mat R12 = R1w * R2w.t();
  cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

  cv::Mat t12x = SkewSymmetricMatrix(t12);

  const cv::Mat &K1 = pKF1->mK;
  const cv::Mat &K2 = pKF2->mK;

  return K1.t().inv() * t12x * R12 * K2.inv();
}

/// @brief 局部范围内剔除 KF
void LocalMapping::KeyFrameCulling() {
  // Check redundant keyframes (only local keyframes)
  // A keyframe is considered redundant if the 90% of the MapPoints it sees, are
  // seen in at least other 3 keyframes (in the same or finer scale) We only
  // consider close stereo points
  vector<KeyFrame *> vpLocalKeyFrames =
      mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();  // 有序共视关键帧向量

  // 遍历共视 KF
  for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(),
                                    vend = vpLocalKeyFrames.end();
       vit != vend; vit++) {
    KeyFrame *pKF = *vit;  // 共视 KF

    // 初始 KF 不剔除
    if (pKF->mnId == 0) continue;
    const vector<MapPoint *> vpMapPoints =
        pKF->GetMapPointMatches();  // 共视 KF MP 观测

    int nObs = 3;  // 观测此 MP 不差于此 KF 的 KF 数量
    // 这个神奇的初始化方法，为什么不直接用字面值初始化常量?
    const int thObs = nObs;          // KF 共视剔除阈值
    int nRedundantObservations = 0;  // 共视 KF 冗余观测数
    int nMPs = 0;                    // MP 观测数

    // 遍历共视 KF MP 观测
    for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
      MapPoint *pMP = vpMapPoints[i];
      if (pMP) {
        if (!pMP->isBad()) {
          // 非单目情况下仅对近双目点进行统计
          if (!mbMonocular) {
            if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
              continue;
          }

          nMPs++;
          if (pMP->Observations() > thObs) {
            const int &scaleLevel = pKF->mvKeysUn[i].octave;  // 尺度
            const map<KeyFrame *, size_t> observations =
                pMP->GetObservations();  // 当前 MP 观测映射
            int nObs = 0;
            for (map<KeyFrame *, size_t>::const_iterator
                     mit = observations.begin(),
                     mend = observations.end();
                 mit != mend; mit++) {
              KeyFrame *pKFi = mit->first;
              if (pKFi == pKF) continue;
              const int &scaleLeveli =
                  pKFi->mvKeysUn[mit->second].octave;  // 在其他关键帧中的尺度

              if (scaleLeveli <= scaleLevel + 1) {
                nObs++;
                if (nObs >= thObs) break;
              }
            }
            if (nObs >= thObs) {
              nRedundantObservations++;
            }
          }
        }
      }
    }

    if (nRedundantObservations > 0.9 * nMPs) pKF->SetBadFlag();
  }
}

/// @brief 向量转反对称矩阵 SO3 hat
/// @param v v3
/// @return v^
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v) {
  return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
          v.at<float>(2), 0, -v.at<float>(0), -v.at<float>(1), v.at<float>(0),
          0);
}

///////////////////////////////////////////////////////// 标志位设置

/// @brief 请求暂停更新地图
void LocalMapping::RequestStop() {
  unique_lock<mutex> lock(mMutexStop);
  mbStopRequested = true;  // 暂停请求标志位
  unique_lock<mutex> lock2(mMutexNewKFs);
  mbAbortBA = true;  // 放弃 BA 优化标志位
}

/// @brief 请求重置
void LocalMapping::RequestReset() {
  {
    unique_lock<mutex> lock(mMutexReset);
    mbResetRequested = true;
  }

  while (1) {
    {
      unique_lock<mutex> lock2(mMutexReset);
      if (!mbResetRequested) break;
    }
    usleep(3000);
  }
}

/// @brief 是否需要暂停
/// @return
bool LocalMapping::Stop() {
  unique_lock<mutex> lock(mMutexStop);
  if (mbStopRequested && !mbNotStop) {
    mbStopped = true;
    cout << "Local Mapping STOP" << endl;
    return true;
  }

  return false;
}

/// @brief 是否已暂停
/// @return
bool LocalMapping::isStopped() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopped;
}

/// @brief 是否有暂停请求
/// @return
bool LocalMapping::stopRequested() {
  unique_lock<mutex> lock(mMutexStop);
  return mbStopRequested;
}

/// @brief 释放队列中 KF
void LocalMapping::Release() {
  unique_lock<mutex> lock(mMutexStop);
  unique_lock<mutex> lock2(mMutexFinish);
  if (mbFinished) return;
  mbStopped = false;
  mbStopRequested = false;
  for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(),
                                  lend = mlNewKeyFrames.end();
       lit != lend; lit++)
    delete *lit;
  mlNewKeyFrames.clear();

  cout << "Local Mapping RELEASE" << endl;
}

/// @brief 是否接收新 KF
/// @return
bool LocalMapping::AcceptKeyFrames() {
  unique_lock<mutex> lock(mMutexAccept);
  return mbAcceptKeyFrames;
}

/// @brief 设置接受关键帧标志位
/// @param flag
void LocalMapping::SetAcceptKeyFrames(bool flag) {
  unique_lock<mutex> lock(mMutexAccept);
  mbAcceptKeyFrames = flag;
}

/// @brief 设置不暂停标志位  Tracking 创建 KF 时调用
/// @param flag
/// @return
bool LocalMapping::SetNotStop(bool flag) {
  unique_lock<mutex> lock(mMutexStop);

  if (flag && mbStopped) return false;

  mbNotStop = flag;

  return true;
}

/// @brief 中断 BA
void LocalMapping::InterruptBA() { mbAbortBA = true; }

/// @brief 重置检查
void LocalMapping::ResetIfRequested() {
  unique_lock<mutex> lock(mMutexReset);
  if (mbResetRequested) {
    mlNewKeyFrames.clear();
    mlpRecentAddedMapPoints.clear();
    mbResetRequested = false;
  }
}

/// @brief 请求终止
void LocalMapping::RequestFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

/// @brief 终止检查
/// @return
bool LocalMapping::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

/// @brief 设置终止
void LocalMapping::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
  unique_lock<mutex> lock2(mMutexStop);
  mbStopped = true;
}

/// @brief 是否已终止
/// @return
bool LocalMapping::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

}  // namespace ORB_SLAM2
