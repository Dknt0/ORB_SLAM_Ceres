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

#include "ORBmatcher.h"

#include <limits.h>
#include <stdint-gcc.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

using namespace std;

namespace ORB_SLAM2 {

const int ORBmatcher::TH_HIGH = 100;  // 匹配点对汉明距离高阈值  相对宽松
const int ORBmatcher::TH_LOW = 50;  // 匹配点对汉明距离低阈值  相对严格
const int ORBmatcher::HISTO_LENGTH = 30;  // 直方图长度

/// @brief 构造函数
/// @param nnratio 最佳匹配与次佳匹配汉明距离比例阈值
/// @param checkOri 是否检查旋转一致性
ORBmatcher::ORBmatcher(float nnratio, bool checkOri)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

/////////////////////////////////////////////////////////////////////////////////
///投影匹配函数

/// @brief 帧与地图点集匹配  Tracking 中追踪 Local Map
///     Tracking::SearchLocalPoints()
/// @param F 当前帧
/// @param vpMapPoints 搜索地图点集
/// @param th 搜索窗口范围倍数
/// @return 匹配点对数量
int ORBmatcher::SearchByProjection(Frame &F,
                                   const vector<MapPoint *> &vpMapPoints,
                                   const float th) {
  int nmatches = 0;  // 匹配数

  const bool bFactor = th != 1.0;  // 阈值不为 1 置 1

  // 遍历地图点集
  for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
    MapPoint *pMP = vpMapPoints[iMP];  // 地图点

    // 如果地图点对于当前帧不可视，忽略
    if (!pMP->mbTrackInView) continue;

    if (pMP->isBad()) continue;

    const int &nPredictedLevel = pMP->mnTrackScaleLevel;  // 金字塔尺度预测级别

    /* 1.在地图点像素平面投影附近，寻找尺度满足要求的候选关键点，窗口范围由视角余弦、特征尺度确定
     */

    // The size of the window will depend on the viewing direction
    float r = RadiusByViewingCos(
        pMP->mTrackViewCos);  // 窗口半径 窗口范围由视角余弦、特征尺度确定

    // 乘阈值
    if (bFactor) r *= th;

    const vector<size_t> vIndices = F.GetFeaturesInArea(
        pMP->mTrackProjX, pMP->mTrackProjY,
        r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1,
        nPredictedLevel);  // 候选关键点序号集
                           // 地图点投影附近的关键点，尺度与预测尺度相同或小一个级别

    if (vIndices.empty()) continue;

    /* 2.计算地图点与候选关键点间汉明距离，寻找最佳、次佳匹配对 */
    const cv::Mat MPdescriptor = pMP->GetDescriptor();  // 地图点描述子

    // 记录最佳匹配对
    int bestDist = 256;
    int bestLevel = -1;
    int bestDist2 = 256;
    int bestLevel2 = -1;
    int bestIdx = -1;

    // Get best and second matches with near keypoints
    // 遍历候选关键点集
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;  // 关键点序号

      // 如果这个关键点有地图点观测，且那个地图点的观测数大于 0，忽略
      // 即，忽略已经观测到有效 MP 的 KP
      if (F.mvpMapPoints[idx])
        if (F.mvpMapPoints[idx]->Observations() > 0) continue;

      // 如果双目点有效，右点坐标误差超过窗口大小，忽略
      if (F.mvuRight[idx] > 0) {
        const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
        if (er > r * F.mvScaleFactors[nPredictedLevel]) continue;
      }

      const cv::Mat &d = F.mDescriptors.row(idx);  // 特征点描述子

      const int dist =
          DescriptorDistance(MPdescriptor, d);  // 地图点与特征点汉明距离

      // 记录最佳、次佳匹配
      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestLevel2 = bestLevel;
        bestLevel = F.mvKeysUn[idx].octave;
        bestIdx = idx;
      } else if (dist < bestDist2) {
        bestLevel2 = F.mvKeysUn[idx].octave;
        bestDist2 = dist;
      }
    }

    /* 3.匹配对筛选，如果匹配距离大于阈值，或最佳、次佳匹配点来自不同层，抛弃匹配对，否则将
     * MP 添加到 F 观测集 */
    // Apply ratio to second match (only if best and second are in the same
    // scale level) 如果最佳匹配距离小于大阈值
    if (bestDist <= TH_HIGH) {
      // 如果最佳匹配和次佳匹配特征点不是来自同一层，忽略
      if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2) continue;

      F.mvpMapPoints[bestIdx] = pMP;  // 记录到帧的观测集
      nmatches++;                     // 匹配数 +1
    }
  }

  return nmatches;
}

/// @brief 匹配帧与相似投影后地图点  Loop Closing 中用于回环检测
///     LoopClosing::ComputeSim3
/// @param pKF 关键帧
/// @param Scw 相似变换矩阵
/// @param vpPoints 候选地图点集
/// @param vpMatched 已匹配的地图点集  按照 KF KP 索引
/// @param th 搜索窗口范围倍数
/// @return 匹配点对数
int ORBmatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw,
                                   const vector<MapPoint *> &vpPoints,
                                   vector<MapPoint *> &vpMatched, int th) {
  // Get Calibration Parameters for later projection
  // 获取内参
  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;

  // Decompose Scw
  cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);      // 带尺度姿态 sRcw
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));  // 尺度 scw
  cv::Mat Rcw = sRcw / scw;                              // 姿态 Rcw
  cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;         // 位置 tcw
  cv::Mat Ow = -Rcw.t() * tcw;  // 相机空间位置 twc

  // Set of MapPoints already found in the KeyFrame
  set<MapPoint *> spAlreadyFound(vpMatched.begin(),
                                 vpMatched.end());  // 已匹配的地图点集合
  spAlreadyFound.erase(static_cast<MapPoint *>(NULL));  // 清除空指针

  int nmatches = 0;  // 匹配数

  // For each Candidate MapPoint Project and Match
  // 遍历候选地图点集
  for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++) {
    MapPoint *pMP = vpPoints[iMP];  // 候选地图点

    // Discard Bad MapPoints and already found
    // 如果候选地图点为坏点，或已经找到匹配，忽略
    if (pMP->isBad() || spAlreadyFound.count(pMP)) continue;

    // Get 3D Coords.
    cv::Mat p3Dw = pMP->GetWorldPos();  // 候选地图点空间位置

    // Transform into Camera Coords.
    cv::Mat p3Dc = Rcw * p3Dw + tcw;  // 候选地图点相对于关键帧位置

    // Depth must be positive
    // 保证深度不为负数
    if (p3Dc.at<float>(2) < 0.0) continue;

    // Project into Image
    // 将点投影到图像平面上
    const float invz = 1 / p3Dc.at<float>(2);
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v)) continue;

    // Depth must be inside the scale invariance region of the point
    const float maxDistance =
        pMP->GetMaxDistanceInvariance();  // 最大尺度无关距离*1.2
    const float minDistance =
        pMP->GetMinDistanceInvariance();  // 最小尺度无关距离*0.8
    cv::Mat PO = p3Dw - Ow;  // 候选地图点相对于关键帧位置在世界系下的表示
                             // 其实和 p3Dc 模长是一样的
    const float dist = cv::norm(PO);  // 距离

    // 判断距离是否在尺度无关范围内
    if (dist < minDistance || dist > maxDistance) continue;

    // Viewing angle must be less than 60 deg
    // 观测角应小于 60 度
    cv::Mat Pn = pMP->GetNormal();  // 平均观测方向

    // cos(60)=0.5
    if (PO.dot(Pn) < 0.5 * dist) continue;

    int nPredictedLevel = pMP->PredictScale(dist, pKF);  // 预测尺度

    // Search in a radius
    const float radius =
        th * pKF->mvScaleFactors[nPredictedLevel];  // 由预测尺度确定窗口半径

    const vector<size_t> vIndices =
        pKF->GetFeaturesInArea(u, v, radius);  // 投影点窗口范围内的候选关键点

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

    int bestDist = 256;
    int bestIdx = -1;
    // 遍历候选关键点
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;  // 关键点 id
      if (vpMatched[idx]) continue;

      const int &kpLevel = pKF->mvKeysUn[idx].octave;  // 关键点尺度级别

      // 如果层数不满足要求，忽略   特征点只能变远一个级别
      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel) continue;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);  // 关键点描述子

      const int dist = DescriptorDistance(dMP, dKF);  // 计算汉明距离

      // 记录最佳匹配和序号
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // 如果匹配距离小于小阈值，保留
    if (bestDist <= TH_LOW) {
      vpMatched[bestIdx] = pMP;
      nmatches++;
    }
  }

  return nmatches;  // 返回匹配点对
}

/// @brief 相邻帧间匹配
///     Tracking::TrackWithMotionModel
/// @param CurrentFrame 当前帧
/// @param LastFrame 上一帧
/// @param th 搜索窗口范围倍数
/// @param bMono 是否为单目
/// @return 上一帧观测地图点与当前帧特征点的匹配点对数量
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                                   const float th, const bool bMono) {
  int nmatches = 0;  // 匹配数量

  // Rotation Histogram (to check rotation consistency)
  // 旋转直方图，用于检查旋转一致性
  vector<int>
      rotHist[HISTO_LENGTH];  // 旋转直方图  [直方图列](vector<特征点序号>)
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;  // 直方图长度的倒数

  /* 1.计算当前帧相对上一帧位姿，对非单目模式，确定当前帧前进或后退 */

  const cv::Mat Rcw =
      CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);  // 当前帧姿态 Rcw
  const cv::Mat tcw =
      CurrentFrame.mTcw.rowRange(0, 3).col(3);  // 当前帧位置 tcw

  const cv::Mat twc = -Rcw.t() * tcw;  // 当前帧位置 twc

  const cv::Mat Rlw =
      LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);  // 上一帧姿态 Rlw
  const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);  // 上一帧位置 tlw

  const cv::Mat tlc = Rlw * twc + tlw;  // 当前帧相对于上一帧位置 tlc

  const bool bForward = (tlc.at<float>(2) > CurrentFrame.mb) &&
                        !bMono;  // 前进标志位  位移 z 大于 0 时  单目无效
  const bool bBackward = (-tlc.at<float>(2) > CurrentFrame.mb) &&
                         !bMono;  // 后退标志位  位移 z 小于 0 时  单目无效

  // 遍历上一帧的观测地图点集
  for (int i = 0; i < LastFrame.N; i++) {
    MapPoint *pMP = LastFrame.mvpMapPoints[i];  // 上一帧的地图点

    // 观测存在，且不为外点
    if (!pMP) continue;
    if (LastFrame.mvbOutlier[i]) continue;

    /* 2.将上一帧观测到的地图点投影到当前帧内，在合理的投影点附近选取尺度、视差满足条件的候选关键点
     */
    // Project
    cv::Mat x3Dw = pMP->GetWorldPos();  // 地图点空间位置
    cv::Mat x3Dc = Rcw * x3Dw + tcw;    // 地图点相对于当前帧位置

    const float xc = x3Dc.at<float>(0);  // 地图点相对于当前帧位置 x
    const float yc = x3Dc.at<float>(1);  // 地图点相对于当前帧位置 y
    const float invzc = 1.0 / x3Dc.at<float>(2);  // 地图点相对于当前帧位置 1/z

    // 深度为负，忽略
    if (invzc < 0) continue;

    float u = CurrentFrame.fx * xc * invzc +
              CurrentFrame.cx;  // 地图点投影当前帧像素平面坐标 u
    float v = CurrentFrame.fy * yc * invzc +
              CurrentFrame.cy;  // 地图点投影当前帧像素平面坐标 v

    // 检查投影点是否位于边界内
    if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX) continue;
    if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY) continue;

    int nLastOctave =
        LastFrame.mvKeys[i].octave;  // 地图点在上一帧内特征点尺度级别

    // Search in a window. Size depends on scale
    float radius =
        th *
        CurrentFrame.mvScaleFactors[nLastOctave];  // 窗口范围由特征尺度决定

    vector<size_t>
        vIndices2;  // 上一帧的地图点在当前帧中候选关键点向量  关键点序号

    // 下面尺度获取时没有使用预测尺度，而是直接使用了上一帧的尺度作为基础，这和其他的重载函数是不同的

    if (bForward)
      // 若前进，取地图点在上一帧内的层数到最高层范围内的关键点
      vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
    else if (bBackward)
      // 若后退，取 0 层到地图点在上一帧内的层数范围内的关键点
      vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
    else
      // 单目时，取层数相差不超过 1 的关键点
      vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave - 1,
                                                 nLastOctave + 1);

    if (vIndices2.empty()) continue;

    /* 3.比较地图点与候选关键点的描述子，选取最佳匹配对 */

    const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

    int bestDist = 256;
    int bestIdx2 = -1;

    // 遍历候选关键点向量，寻找最佳匹配对
    for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                        vend = vIndices2.end();
         vit != vend; vit++) {
      const size_t i2 = *vit;  // 关键点序号

      // 如果这个关键点已经存在地图点观测，忽略
      if (CurrentFrame.mvpMapPoints[i2])
        if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0) continue;

      // 对于双目点，保证投影后视差误差不超过窗口半径
      if (CurrentFrame.mvuRight[i2] > 0) {
        const float ur = u - CurrentFrame.mbf * invzc;  // 投影右点坐标
        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
        if (er > radius) continue;
      }

      const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);  // 关键点描述子

      const int dist = DescriptorDistance(dMP, d);  // 计算汉明距离

      // 记录最佳匹配关键点序号、匹配误差
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx2 = i2;
      }
    }

    // 匹配误差应小于大阈值
    if (bestDist > TH_HIGH) continue;

    CurrentFrame.mvpMapPoints[bestIdx2] = pMP;  // 记录到当前帧的观测集
    nmatches++;                                 // 匹配数 +1

    /* 4.根据点对在当前帧、上一帧内特征主方向之差，将匹配对记录到旋转直方图中 */

    // 如果需要检测旋转一致性
    if (mbCheckOrientation) {
      float rot = LastFrame.mvKeysUn[i].angle -
                  CurrentFrame.mvKeysUn[bestIdx2]
                      .angle;  // 上一帧关键点相对于当前帧特征旋转角度
      // 角度归一化  0~360
      if (rot < 0.0) rot += 360.0f;
      // 直方图预留了 30 列，但这里最大列号 round(360 / 30) = 12，貌似只有 0~12
      // 列会有点落入?
      int bin = round(rot * factor);  // 直方图序号
      if (bin == HISTO_LENGTH) bin = 0;
      assert(bin >= 0 && bin < HISTO_LENGTH);
      rotHist[bin].push_back(bestIdx2);
    }
  }

  /* 5.旋转一致性检验，仅保留直方图点数最多三个列中的匹配对 */
  // Apply rotation consistency
  // 如果需要检测旋转一致性
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2,
                       ind3);  // 寻找直方图中前三大的列标号

    // 遍历直方图  第 i 列
    for (int i = 0; i < HISTO_LENGTH; i++) {
      // 对于前三之外的列
      if (i != ind1 && i != ind2 && i != ind3) {
        // 遍历区域中地图点观测  关键点 j
        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
          CurrentFrame.mvpMapPoints[rotHist[i][j]] =
              static_cast<MapPoint *>(NULL);  // 清除观测
          nmatches--;                         // 匹配数-1
        }
      }
    }
  }

  return nmatches;  // 返回匹配点对数
}

/// @brief 匹配帧与关键帧  sAlreadyFound 中的 MP 不参与匹配  Tracking
/// 中用于重定位
///     Tracking::Relocalization
/// @param CurrentFrame 当前帧
/// @param pKF 关键帧
/// @param sAlreadyFound 已经找到的地图点
/// @param th 搜索窗口范围倍数
/// @param ORBdist 描述子距离阈值
/// @return
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF,
                                   const set<MapPoint *> &sAlreadyFound,
                                   const float th, const int ORBdist) {
  int nmatches = 0;  // 匹配数

  const cv::Mat Rcw =
      CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);  // 当前帧姿态 Rcw
  const cv::Mat tcw =
      CurrentFrame.mTcw.rowRange(0, 3).col(3);  // 当前帧位置 tcw
  const cv::Mat Ow = -Rcw.t() * tcw;            // 当前帧空间位置 twc

  // Rotation Histogram (to check rotation consistency)
  vector<int> rotHist[HISTO_LENGTH];  // 旋转直方图
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  const vector<MapPoint *> vpMPs =
      pKF->GetMapPointMatches();  // 获取关键帧观测地图点集

  // 遍历关键帧观测地图点集
  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    MapPoint *pMP = vpMPs[i];  // 地图点

    // 如果地图点观测存在，且不为坏点，且没有被找到
    if (pMP) {
      if (!pMP->isBad() && !sAlreadyFound.count(pMP)) {
        /* 1.将地图点投影到 Fc 图像平面上 */
        // Project
        cv::Mat x3Dw = pMP->GetWorldPos();  // 地图点位置 tw
        cv::Mat x3Dc = Rcw * x3Dw + tcw;  // 地图点相对于关键帧位置 tc

        // 计算地图点在关键帧内像素坐标
        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);

        const float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
        const float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX) continue;
        if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY) continue;

        /* 2.预测特征尺度，按照预测尺度确定窗口半径 */

        // Compute predicted scale level
        cv::Mat PO = x3Dw - Ow;
        float dist3D = cv::norm(PO);  // 距离

        const float maxDistance =
            pMP->GetMaxDistanceInvariance();  // 最大尺度无关距离*1.2
        const float minDistance =
            pMP->GetMinDistanceInvariance();  // 最小尺度无关距离*0.8

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance) continue;

        int nPredictedLevel =
            pMP->PredictScale(dist3D, &CurrentFrame);  // 预测尺度

        // Search in a window
        const float radius =
            th *
            CurrentFrame
                .mvScaleFactors[nPredictedLevel];  // 窗口范围由预测特征尺度决定

        const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(
            u, v, radius, nPredictedLevel - 1,
            nPredictedLevel +
                1);  // 候选关键点序号集
                     // 地图点投影附近的关键点，尺度与预测尺度相差不超过1

        if (vIndices2.empty()) continue;

        const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

        int bestDist = 256;
        int bestIdx2 = -1;

        /* 3.遍历窗口中特征点，记录最佳匹配 */

        // 遍历候选关键点
        for (vector<size_t>::const_iterator vit = vIndices2.begin();
             vit != vIndices2.end(); vit++) {
          const size_t i2 = *vit;  // 关键点序号
          if (CurrentFrame.mvpMapPoints[i2]) continue;

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);  // 关键点描述子

          const int dist = DescriptorDistance(dMP, d);  // 汉明距离

          // 记录最佳距离和序号
          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        /* 4.如果最佳匹配距离小于阈值，为 KP 记录这个 MP，添加 KP
         * 对应关系到旋转直方图中 */

        // 如果最佳距离小于给定的描述子阈值
        if (bestDist <= ORBdist) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;

          // 如果检查姿态一致性
          if (mbCheckOrientation) {
            float rot =
                pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
            if (rot < 0.0) rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH) bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
          }
        }
      }
    }
  }

  /* 旋转一致性检测 */

  // 如果需要检测旋转一致性
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2,
                       ind3);  // 寻找直方图中前三大的列标号
    // 遍历直方图  第 i 列
    for (int i = 0; i < HISTO_LENGTH; i++) {
      // 对于前三之外的列
      if (i != ind1 && i != ind2 && i != ind3) {
        // 遍历区域中地图点观测  关键点 j
        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
          CurrentFrame.mvpMapPoints[rotHist[i][j]] = NULL;  // 清除观测
          nmatches--;                                       // 匹配数-1
        }
      }
    }
  }

  return nmatches;
}

/// @brief Sim3 互投影匹配 给出帧间相似变换，搜索两个关键帧的匹配点对
///     LoopClosing::ComputeSim3
/// @param pKF1 关键帧1
/// @param pKF2 关键帧2
/// @param vpMatches12 已匹配的地图点 按照 KF1 关键点索引 结果也存放于这个向量中
/// @param s12 2 相对 1 尺度
/// @param R12 2 相对 1 姿态
/// @param t12 2 相对 1 位置
/// @param th 搜索窗口范围倍数
/// @return 匹配点对数
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2,
                             vector<MapPoint *> &vpMatches12, const float &s12,
                             const cv::Mat &R12, const cv::Mat &t12,
                             const float th) {
  // 关键帧 1 内参
  const float &fx = pKF1->fx;  // 关键帧 1 fx
  const float &fy = pKF1->fy;  // 关键帧 1 fy
  const float &cx = pKF1->cx;  // 关键帧 1 cx
  const float &cy = pKF1->cy;  // 关键帧 1 cy

  // Camera 1 from world
  cv::Mat R1w = pKF1->GetRotation();     // 关键帧 1 姿态 R1w
  cv::Mat t1w = pKF1->GetTranslation();  // 关键帧 1 位置 t1w

  // Camera 2 from world
  cv::Mat R2w = pKF2->GetRotation();     // 关键帧 2 姿态 R2w
  cv::Mat t2w = pKF2->GetTranslation();  // 关键帧 2 位置 t2w

  // Transformation between cameras
  cv::Mat sR12 = s12 * R12;              // 2 相对 1 sR12
  cv::Mat sR21 = (1.0 / s12) * R12.t();  // 1 相对 2 sR21
  cv::Mat t21 = -sR21 * t12;             // 1 相对 2 位置  带尺度?

  const vector<MapPoint *> vpMapPoints1 =
      pKF1->GetMapPointMatches();      // 关键帧 1 匹配地图点集
  const int N1 = vpMapPoints1.size();  // 关键帧 1 关键点数

  const vector<MapPoint *> vpMapPoints2 =
      pKF2->GetMapPointMatches();      // 关键帧 2 匹配地图点集
  const int N2 = vpMapPoints2.size();  // 关键帧 2 关键点数

  vector<bool> vbAlreadyMatched1(N1, false);  // 关键帧 1 特征点已匹配标志位
  vector<bool> vbAlreadyMatched2(N2, false);  // 关键帧 2 特征点已匹配标志位

  // 遍历关键帧 1 已经匹配的地图点
  for (int i = 0; i < N1; i++) {
    MapPoint *pMP = vpMatches12[i];  // 关键帧 1 地图点
    // 如果存在地图点观测
    if (pMP) {
      vbAlreadyMatched1[i] = true;  // 标志位真
      int idx2 =
          pMP->GetIndexInKeyFrame(pKF2);  // 查询地图点在关键帧 2 中关键点索引
      if (idx2 >= 0 && idx2 < N2)
        vbAlreadyMatched2[idx2] = true;  // 若是，标志位真
    }
  }

  vector<int> vnMatch1(N1, -1);  // KF1 关键点在 KF2 中的匹配关键点
  vector<int> vnMatch2(N2, -1);  // 关键帧 2 地图点匹配？

  // Transform from KF1 to KF2 and search
  // 遍历关键帧 1 地图点集，将 KF1 的地图点变换到 KF2 中搜索匹配
  for (int i1 = 0; i1 < N1; i1++) {
    MapPoint *pMP = vpMapPoints1[i1];  // 关键帧 1 地图点

    // 若不存在观测，或已匹配，或为坏点，忽略
    if (!pMP || vbAlreadyMatched1[i1]) continue;
    if (pMP->isBad()) continue;

    cv::Mat p3Dw = pMP->GetWorldPos();  // 关键帧 1 地图点空间位置
    cv::Mat p3Dc1 = R1w * p3Dw + t1w;  // 关键帧 1 地图点相对于关键帧 1 位置
    cv::Mat p3Dc2 = sR21 * p3Dc1 + t21;  // 关键帧 1 地图点相对于关键帧 2 位置

    // Depth must be positive
    // 深度必须为正
    if (p3Dc2.at<float>(2) < 0.0) continue;

    // 计算在 KF2 中投影像素坐标
    const float invz = 1.0 / p3Dc2.at<float>(2);
    const float x = p3Dc2.at<float>(0) * invz;
    const float y = p3Dc2.at<float>(1) * invz;
    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    // 点应在图像边界内
    if (!pKF2->IsInImage(u, v)) continue;

    const float maxDistance =
        pMP->GetMaxDistanceInvariance();  // 最大尺度无关距离*1.2
    const float minDistance =
        pMP->GetMinDistanceInvariance();   // 最小尺度无关距离*0.8
    const float dist3D = cv::norm(p3Dc2);  // 距离

    // Depth must be inside the scale invariance region
    // 距离应在尺度无关范围内
    if (dist3D < minDistance || dist3D > maxDistance) continue;

    // Compute predicted octave
    const int nPredictedLevel =
        pMP->PredictScale(dist3D, pKF2);  // 预测金字塔尺度级别

    // Search in a radius
    const float radius =
        th * pKF2->mvScaleFactors[nPredictedLevel];  // 依据尺度级别确定窗口半径

    const vector<size_t> vIndices = pKF2->GetFeaturesInArea(
        u, v, radius);  // 获取 KF2 投影点窗口范围内候选关键点

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

    int bestDist = INT_MAX;
    int bestIdx = -1;
    // 遍历候选关键点，寻找最佳匹配
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;  // 关键点序号

      const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];  // 关键点

      // 判断尺度是否合理   只能不变或变远
      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF2->mDescriptors.row(idx);  // 关键点描述子

      const int dist = DescriptorDistance(dMP, dKF);  // 汉明距离

      // 记录最佳匹配距离和序号
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // 如果小于大阈值，记录
    if (bestDist <= TH_HIGH) {
      vnMatch1[i1] = bestIdx;
    }
  }

  // Transform from KF2 to KF1 and search
  // 遍历 KF2 地图点  投影到 KF1 中寻找匹配
  for (int i2 = 0; i2 < N2; i2++) {
    MapPoint *pMP = vpMapPoints2[i2];  // 关键帧 2 地图点

    // 若不存在观测，或已匹配，或为坏点，忽略
    if (!pMP || vbAlreadyMatched2[i2]) continue;
    if (pMP->isBad()) continue;

    cv::Mat p3Dw = pMP->GetWorldPos();  // 关键帧 2 地图点空间位置
    cv::Mat p3Dc2 = R2w * p3Dw + t2w;  // 关键帧 2 地图点相对于关键帧 2 位置
    cv::Mat p3Dc1 = sR12 * p3Dc2 + t12;  // 关键帧 2 地图点相对于关键帧 1 位置

    // Depth must be positive
    // 深度必须为正
    if (p3Dc1.at<float>(2) < 0.0) continue;

    // 计算在 KF1 中投影像素坐标
    const float invz = 1.0 / p3Dc1.at<float>(2);
    const float x = p3Dc1.at<float>(0) * invz;
    const float y = p3Dc1.at<float>(1) * invz;
    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    // 点应在图像边界内
    if (!pKF1->IsInImage(u, v)) continue;

    const float maxDistance =
        pMP->GetMaxDistanceInvariance();  // 最大尺度无关距离*1.2
    const float minDistance =
        pMP->GetMinDistanceInvariance();   // 最小尺度无关距离*0.8
    const float dist3D = cv::norm(p3Dc1);  // 距离

    // Depth must be inside the scale pyramid of the image
    // 距离应在尺度无关范围内
    if (dist3D < minDistance || dist3D > maxDistance) continue;

    // Compute predicted octave
    const int nPredictedLevel =
        pMP->PredictScale(dist3D, pKF1);  // 预测金字塔尺度级别

    // Search in a radius of 2.5*sigma(ScaleLevel)
    const float radius = th * pKF1->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF1->GetFeaturesInArea(
        u, v, radius);  // 获取 KF1 投影点窗口范围内候选关键点

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

    int bestDist = INT_MAX;
    int bestIdx = -1;
    // 遍历候选关键点，寻找最佳匹配
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;  // 关键点序号

      const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

      // 判断尺度是否合理   只能不变或变远
      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF1->mDescriptors.row(idx);  // 关键点描述子

      const int dist = DescriptorDistance(dMP, dKF);  // 汉明距离

      // 记录最佳匹配距离和序号
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // 如果小于大阈值，记录
    if (bestDist <= TH_HIGH) {
      vnMatch2[i2] = bestIdx;
    }
  }

  // Check agreement
  // 检查互投影匹配结果
  int nFound = 0;  // 匹配数
  // 遍历 KF1 匹配结果
  for (int i1 = 0; i1 < N1; i1++) {
    int idx2 = vnMatch1[i1];  // KF1 匹配到 KF2 中的关键点序号

    if (idx2 >= 0) {
      int idx1 = vnMatch2[idx2];  // KF2 中的这个关键点匹配回 KF1 中的关键序号
                                  // 和 i1 应该相等
      if (idx1 == i1) {
        // 如果相等，代表匹配一致
        vpMatches12[i1] = vpMapPoints2[idx2];  // 存放结果
        nFound++;
      }
    }
  }

  return nFound;
}

/////////////////////////////////////////////////////////////////////////////////
///词袋搜索函数

/// @brief 搜索关键帧地图点和帧关键点的匹配  基于视觉词汇约束的暴力匹配
///     Tracking::TrackReferenceKeyFrame
///     Tracking::Relocalization
/// @param pKF 关键帧  作为 const 使用
/// @param F 帧
/// @param vpMapPointMatches 帧地图点匹配 按 F 关键点索引  输出
/// @return 匹配点对数
int ORBmatcher::SearchByBoW(KeyFrame *pKF, Frame &F,
                            vector<MapPoint *> &vpMapPointMatches) {
  const vector<MapPoint *> vpMapPointsKF =
      pKF->GetMapPointMatches();  // 关键帧地图点观测集 const

  vpMapPointMatches = vector<MapPoint *>(
      F.N, static_cast<MapPoint *>(NULL));  // 清空帧地图点匹配集

  const DBoW2::FeatureVector &vFeatVecKF =
      pKF->mFeatVec;  // KF 视觉特征向量 const   map<节点id, 关键点序号集
                      // vector<int>>

  int nmatches = 0;  // 匹配点对数

  // 旋转直方图
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  // We perform the matching over ORB that belong to the same vocabulary node
  // (at a certain level)
  DBoW2::FeatureVector::const_iterator KFit =
      vFeatVecKF.begin();  // KF 特征向量迭代器
  DBoW2::FeatureVector::const_iterator Fit =
      F.mFeatVec.begin();  // F 特征向量迭代器
  DBoW2::FeatureVector::const_iterator KFend =
      vFeatVecKF.end();  // KF 特征向量末尾迭代器
  DBoW2::FeatureVector::const_iterator Fend =
      F.mFeatVec.end();  // F 特征向量末尾迭代器

  // 遍历关键帧特征向量与帧特征向量，在两帧位于相同视觉词汇节点的关键点中，寻找匹配
  // 视觉词汇约束下的暴力匹配
  while (KFit != KFend && Fit != Fend) {
    // 如果位于同一个视觉词汇节点
    if (KFit->first == Fit->first) {
      const vector<unsigned int> vIndicesKF =
          KFit->second;  // KF 候选关键点序号集
      const vector<unsigned int> vIndicesF = Fit->second;  // F 候选关键点序号集

      // 遍历 KF 候选关键点对应的地图点
      for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
        const unsigned int realIdxKF = vIndicesKF[iKF];  // KF 关键点 id

        MapPoint *pMP = vpMapPointsKF[realIdxKF];  // KF 关键点对应地图点

        // 如果地图点存在且不为坏点，继续
        if (!pMP) continue;
        if (pMP->isBad()) continue;

        const cv::Mat &dKF =
            pKF->mDescriptors.row(realIdxKF);  // KF 关键点描述子

        int bestDist1 = 256;
        int bestIdxF = -1;
        int bestDist2 = 256;

        // 遍历 F 候选关键点  记录最佳、次佳匹配距离
        for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
          const unsigned int realIdxF = vIndicesF[iF];  // F 关键点 id

          // 如果已经匹配，忽略
          if (vpMapPointMatches[realIdxF]) continue;

          const cv::Mat &dF = F.mDescriptors.row(realIdxF);  // F 关键点描述子

          const int dist = DescriptorDistance(dKF, dF);  // 描述子距离

          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdxF = realIdxF;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        // 如果最佳距离小于小阈值
        if (bestDist1 <= TH_LOW) {
          // 如果最佳匹配远优于次佳匹配，记录之
          if (static_cast<float>(bestDist1) <
              mfNNratio * static_cast<float>(bestDist2)) {
            vpMapPointMatches[bestIdxF] = pMP;  // 记录匹配结果

            const cv::KeyPoint &kp =
                pKF->mvKeysUn[realIdxKF];  // 关键帧关键点  const

            // 如果需要检查姿态一致性，记录旋转直方图
            if (mbCheckOrientation) {
              float rot = kp.angle - F.mvKeys[bestIdxF].angle;
              if (rot < 0.0) rot += 360.0f;
              int bin = round(rot * factor);
              if (bin == HISTO_LENGTH) bin = 0;
              assert(bin >= 0 && bin < HISTO_LENGTH);
              rotHist[bin].push_back(bestIdxF);
            }
            nmatches++;  // 匹配数 +1
          }
        }
      }

      // 迭代器前移
      KFit++;
      Fit++;
    } else if (KFit->first < Fit->first) {
      KFit = vFeatVecKF.lower_bound(Fit->first);  // 依据键值寻找映射对
    } else {
      Fit = F.mFeatVec.lower_bound(KFit->first);  // 依据键值寻找映射对
    }
  }

  // 检查姿态一致性
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2,
                       ind3);  // 寻找直方图前三大的列序号

    for (int i = 0; i < HISTO_LENGTH; i++) {
      // 抛弃前三以外列中的匹配
      if (i == ind1 || i == ind2 || i == ind3) continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vpMapPointMatches[rotHist[i][j]] =
            static_cast<MapPoint *>(NULL);  // 清除匹配
        nmatches--;
      }
    }
  }

  return nmatches;  // 匹配点对
}

/// @brief 搜索 KF1 KP 和 KF2 KP 的匹配  只匹配关联到 MP 的 KP
/// 基于视觉词汇约束暴力匹配  用于重定位和回环检测
///     LoopClosing::ComputeSim3
/// @param pKF1 关键帧 1
/// @param pKF2 关键帧 2
/// @param vpMatches12 地图点匹配   KF1 MP 匹配到 KF2 MP  按 KF1 关键点索引
/// @return 匹配点对数
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2,
                            vector<MapPoint *> &vpMatches12) {
  // KF1
  const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;  // KF1 关键点
  const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;  // KF1 视觉特征向量
  const vector<MapPoint *> vpMapPoints1 =
      pKF1->GetMapPointMatches();                    // KF1 地图点
  const cv::Mat &Descriptors1 = pKF1->mDescriptors;  // KF1 关键点描述子

  // KF2
  const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;  // KF2 关键点
  const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;  // KF2 视觉特征向量
  const vector<MapPoint *> vpMapPoints2 =
      pKF2->GetMapPointMatches();                    // KF2 地图点
  const cv::Mat &Descriptors2 = pKF2->mDescriptors;  // KF2 关键点描述子

  vpMatches12 =
      vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));
  vector<bool> vbMatched2(vpMapPoints2.size(),
                          false);  // KF2 地图点已被匹配  标志位

  // 旋转直方图
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);

  const float factor = 1.0f / HISTO_LENGTH;

  int nmatches = 0;  // 匹配点对数

  DBoW2::FeatureVector::const_iterator f1it =
      vFeatVec1.begin();  // KF1 视觉特征向量迭代器
  DBoW2::FeatureVector::const_iterator f2it =
      vFeatVec2.begin();  // KF2 视觉特征向量迭代器
  DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
  DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

  // 遍历 KF1 视觉特征向量与 KF2
  // 视觉特征向量，在两帧位于相同视觉词汇节点的关键点中，寻找匹配
  // 视觉词汇约束下的暴力匹配
  while (f1it != f1end && f2it != f2end) {
    // 如果位于同一个视觉词汇节点
    if (f1it->first == f2it->first) {
      // 遍历 KF1 视觉节点地图点
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        const size_t idx1 = f1it->second[i1];  // KF1 关键点 id

        MapPoint *pMP1 = vpMapPoints1[idx1];  // KF1 地图点

        // 匹配存在且不为坏点
        if (!pMP1) continue;
        if (pMP1->isBad()) continue;

        const cv::Mat &d1 = Descriptors1.row(idx1);  // KF1 关键点描述子

        int bestDist1 = 256;
        int bestIdx2 = -1;
        int bestDist2 = 256;

        // 遍历 KF2 视觉节点地图点  记录最佳、次佳匹配
        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          const size_t idx2 = f2it->second[i2];  // KF2 关键点 id

          MapPoint *pMP2 = vpMapPoints2[idx2];  // KF2 地图点

          // 如果地图点已被匹配、地图点观测不存在、地图点为坏点，忽略
          if (vbMatched2[idx2] || !pMP2) continue;
          if (pMP2->isBad()) continue;

          const cv::Mat &d2 = Descriptors2.row(idx2);  // KF2 关键点描述子

          int dist = DescriptorDistance(d1, d2);  // 计算汉明距离

          // 记录最佳、次佳距离，匹配序号
          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdx2 = idx2;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        // 如果最佳距离小于低阈值
        if (bestDist1 < TH_LOW) {
          // 如果最佳匹配远优于次佳匹配
          if (static_cast<float>(bestDist1) <
              mfNNratio * static_cast<float>(bestDist2)) {
            vpMatches12[idx1] =
                vpMapPoints2[bestIdx2];   // 为 KF1 记录这个地图点
            vbMatched2[bestIdx2] = true;  // KF2 地图点记录为已经匹配

            // 如果检查姿态一致性
            if (mbCheckOrientation) {
              float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
              if (rot < 0.0) rot += 360.0f;
              int bin = round(rot * factor);
              if (bin == HISTO_LENGTH) bin = 0;
              assert(bin >= 0 && bin < HISTO_LENGTH);
              rotHist[bin].push_back(idx1);
            }
            nmatches++;
          }
        }
      }
      // 迭代器前移
      f1it++;
      f2it++;
    } else if (f1it->first < f2it->first) {
      f1it = vFeatVec1.lower_bound(f2it->first);  // 依据键值寻找映射对
    } else {
      f2it = vFeatVec2.lower_bound(f1it->first);  // 依据键值寻找映射对
    }
  }

  // 检查姿态一致性
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2,
                       ind3);  // 寻找直方图前三大的列序号

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3) continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vpMatches12[rotHist[i][j]] = static_cast<MapPoint *>(NULL);  // 清除匹配
        nmatches--;
      }
    }
  }

  return nmatches;  // 匹配点对
}

/////////////////////////////////////////////////////////////////////////////////
///初始化与三角化

/// @brief 初始化匹配  仅用于单目
/// @param F1 初始化参考帧
/// @param F2 当前帧
/// @param vbPrevMatched F1 到 F2 预匹配  按 F1 关键点索引
/// 匹配结果添加到这个向量中
/// @param vnMatches12 F1 匹配到 F2 关键点序号  按 F1 关键点索引
/// @param windowSize 窗口大小
/// @return 匹配点数
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2,
                                        vector<cv::Point2f> &vbPrevMatched,
                                        vector<int> &vnMatches12,
                                        int windowSize) {
  int nmatches = 0;  // 匹配点对数
  vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

  // 旋转直方图
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  vector<int> vMatchedDistance(F2.mvKeysUn.size(),
                               INT_MAX);  // 匹配距离  F2 KP 索引
  vector<int> vnMatches21(F2.mvKeysUn.size(), -1);  // F2 到 F1 匹配  F2 KP 索引

  // 遍历 F1 中位于 0 层的关键点
  for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++) {
    cv::KeyPoint kp1 = F1.mvKeysUn[i1];  // F1 关键点
    int level1 = kp1.octave;             // F1 关键点层数
    // 层数大于 0 忽略
    if (level1 > 0) continue;

    // 获取在预匹配位置附近窗口内的 KP 作为候选
    vector<size_t> vIndices2 = F2.GetFeaturesInArea(
        vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1,
        level1);  // 获取 F1 KP 在 F2 预匹配位置附近的候选关键点

    if (vIndices2.empty()) continue;

    cv::Mat d1 = F1.mDescriptors.row(i1);  // F1 KP 描述子

    int bestDist = INT_MAX;
    int bestDist2 = INT_MAX;
    int bestIdx2 = -1;

    // 遍历 F2 候选关键点，寻找最佳、次佳匹配
    for (vector<size_t>::iterator vit = vIndices2.begin();
         vit != vIndices2.end(); vit++) {
      size_t i2 = *vit;  // F2 KP id

      cv::Mat d2 = F2.mDescriptors.row(i2);  // F2 KP 描述子

      int dist = DescriptorDistance(d1, d2);  // 计算汉明距离

      // 如果 F2 KP 已经有更好的匹配，忽略
      if (vMatchedDistance[i2] <= dist) continue;

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestIdx2 = i2;
      } else if (dist < bestDist2) {
        bestDist2 = dist;
      }
    }

    // 如果最佳匹配小于小阈值
    if (bestDist <= TH_LOW) {
      // 如果最佳匹配远优于次佳匹配
      if (bestDist < (float)bestDist2 * mfNNratio) {
        // 如果这一次匹配结果替换了之前的结果，序号清除之前的匹配
        if (vnMatches21[bestIdx2] >= 0) {
          vnMatches12[vnMatches21[bestIdx2]] = -1;  // 清除之前 F1 到 F2 的匹配
          nmatches--;
        }
        vnMatches12[i1] = bestIdx2;
        vnMatches21[bestIdx2] = i1;
        vMatchedDistance[bestIdx2] = bestDist;
        nmatches++;

        // 检查姿态一致性
        if (mbCheckOrientation) {
          float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
          if (rot < 0.0) rot += 360.0f;
          int bin = round(rot * factor);
          if (bin == HISTO_LENGTH) bin = 0;
          assert(bin >= 0 && bin < HISTO_LENGTH);
          rotHist[bin].push_back(i1);
        }
      }
    }
  }

  // 检查姿态一致性
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3) continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        int idx1 = rotHist[i][j];
        if (vnMatches12[idx1] >= 0) {
          vnMatches12[idx1] = -1;
          nmatches--;
        }
      }
    }
  }

  // Update prev matched
  //  遍历 F1 到 F2 关键点匹配集   更新F1 到 F2 预匹配
  for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
    if (vnMatches12[i1] >= 0)
      vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

  return nmatches;
}

/// @brief 三角化匹配  检查对极约束的词袋匹配
///     LocalMapping::CreateNewMapPoints()
/// @param pKF1 关键帧 1  当前 KF
/// @param pKF2 关键帧 2  近邻 KF
/// @param F12 基本矩阵
/// @param vMatchedPairs 匹配点对  vector<pair<KF1 KP idx, KF2 KP idx>>
/// @param bOnlyStereo 仅双目点标志位
/// @return 匹配点对数
int ORBmatcher::SearchForTriangulation(
    KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
    vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo) {
  const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;  // KF1 视觉特征向量
  const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;  // KF2 视觉特征向量

  // Compute epipole in second image
  cv::Mat Cw = pKF1->GetCameraCenter();       // KF1 t1wc
  cv::Mat R2w = pKF2->GetRotation();          // KF2 R2cw
  cv::Mat t2w = pKF2->GetTranslation();       // KF2 t2cw
  cv::Mat C2 = R2w * Cw + t2w;                // KF1 在 KF2 下位置  t1c2
  const float invz = 1.0f / C2.at<float>(2);  // 逆深度
  const float ex = pKF2->fx * C2.at<float>(0) * invz +
                   pKF2->cx;  // KF1 在 KF2 图像平面投影点  极点  u
  const float ey = pKF2->fy * C2.at<float>(1) * invz +
                   pKF2->cy;  // KF1 在 KF2 图像平面投影点  极点  v

  // Find matches between not tracked keypoints
  // Matching speed-up by ORB Vocabulary
  // Compare only ORB that share the same node

  int nmatches = 0;                         // 匹配点数
  vector<bool> vbMatched2(pKF2->N, false);  // KF2 特征点已被匹配 标志位
  vector<int> vMatches12(
      pKF1->N, -1);  // KF1 到 KF2 的匹配  按照 KF1 KP 索引  记录 KF2 特征点序号

  // 旋转直方图
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);

  const float factor = 1.0f / HISTO_LENGTH;

  DBoW2::FeatureVector::const_iterator f1it =
      vFeatVec1.begin();  // KF1 视觉特征向量迭代器
  DBoW2::FeatureVector::const_iterator f2it =
      vFeatVec2.begin();  // KF2 视觉特征向量迭代器
  DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
  DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

  // 遍历 KF1 视觉特征向量与 KF2
  // 视觉特征向量，在两帧位于相同视觉词汇节点的关键点中，寻找匹配
  // 视觉词汇约束下的暴力匹配
  while (f1it != f1end && f2it != f2end) {
    // 如果位于同一个视觉词汇节点
    if (f1it->first == f2it->first) {
      // 遍历 KF1 视觉节点地图点
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        const size_t idx1 = f1it->second[i1];  // KF1 关键点 id

        MapPoint *pMP1 = pKF1->GetMapPoint(idx1);  // KF1 地图点

        // If there is already a MapPoint skip
        //// 如果 KF1 KP 已经匹配到 MP 跳过
        if (pMP1) continue;

        // 如果深度有效，置位
        const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;

        if (bOnlyStereo)
          if (!bStereo1) continue;

        const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];  // KF1 KP

        const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);  // KF1 KP 描述子

        int bestDist = TH_LOW;
        int bestIdx2 = -1;  // 最佳 KF2 KP

        // 遍历 KF2 视觉节点地图点  记录最佳、次佳匹配
        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          size_t idx2 = f2it->second[i2];  // KF2 关键点 id

          MapPoint *pMP2 = pKF2->GetMapPoint(idx2);  // KF2 MP

          // If we have already matched or there is a MapPoint skip
          // 如果 KF2 KP 匹配存在或已经匹配到 MP，跳过
          if (vbMatched2[idx2] || pMP2) continue;

          // 如果深度有效，置位
          const bool bStereo2 = pKF2->mvuRight[idx2] >= 0;

          if (bOnlyStereo)
            if (!bStereo2) continue;

          const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);  // KF2 关键点描述子

          const int dist = DescriptorDistance(d1, d2);  // 计算汉明距离

          if (dist > TH_LOW || dist > bestDist) continue;

          const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];  // KF2 KP

          // 如果两个 KP 都不是双目点，进行三角化
          if (!bStereo1 && !bStereo2) {
            // 极线
            const float distex = ex - kp2.pt.x;
            const float distey = ey - kp2.pt.y;
            // 极线长度小于阈值(0层为10，高层更大)，忽略
            if (distex * distex + distey * distey <
                100 * pKF2->mvScaleFactors[kp2.octave])
              continue;
          }

          // 特征匹配极线距离检查
          if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2)) {
            bestIdx2 = idx2;
            bestDist = dist;
          }
        }

        if (bestIdx2 >= 0) {
          const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
          vMatches12[idx1] = bestIdx2;  // 记录匹配
          nmatches++;

          // 姿态一致性检验
          if (mbCheckOrientation) {
            float rot = kp1.angle - kp2.angle;
            if (rot < 0.0) rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH) bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(idx1);
          }
        }
      }

      f1it++;
      f2it++;
    } else if (f1it->first < f2it->first) {
      f1it = vFeatVec1.lower_bound(f2it->first);
    } else {
      f2it = vFeatVec2.lower_bound(f1it->first);
    }
  }

  // 姿态一致性检验
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3) continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vMatches12[rotHist[i][j]] = -1;
        nmatches--;
      }
    }
  }

  vMatchedPairs.clear();  // 清空匹配对
  vMatchedPairs.reserve(nmatches);

  // 遍历 KF1 关键点   记录对 KF2 关键点的匹配结果
  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (vMatches12[i] < 0) continue;
    vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
  }

  return nmatches;
}

/////////////////////////////////////////////////////////////////////////////////
///地图点融合

/// @brief 匹配地图点与关键帧，融合重复地图点  局部建图地图点融合
///     LocalMapping::SearchInNeighbors()
/// @param pKF 关键帧
/// @param vpMapPoints 近邻地图点集
/// @param th 搜索窗口范围倍数
/// @return 添加、融合的地图点数量
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints,
                     const float th) {
  cv::Mat Rcw = pKF->GetRotation();     // 关键帧姿态 Rcw
  cv::Mat tcw = pKF->GetTranslation();  // 关键帧位置 tcw

  // 内参
  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;
  const float &bf = pKF->mbf;

  cv::Mat Ow = pKF->GetCameraCenter();  // KF 位置 twc

  int nFused = 0;  // 新添加、融合的地图点

  const int nMPs = vpMapPoints.size();  // 近邻地图点总数

  /* 寻找 MP 与 KF 的匹配 */
  // 遍历地图点，寻找匹配关键点
  for (int i = 0; i < nMPs; i++) {
    MapPoint *pMP = vpMapPoints[i];  // 近邻地图点

    // 如果地图点观测有效、不为坏点，且没有被关键帧观测到
    if (!pMP) continue;
    if (pMP->isBad() || pMP->IsInKeyFrame(pKF)) continue;

    cv::Mat p3Dw = pMP->GetWorldPos();  // 地图点空间位置
    cv::Mat p3Dc = Rcw * p3Dw + tcw;    // KP 相对于 KF 位置

    // Depth must be positive
    // 检查深度为正
    if (p3Dc.at<float>(2) < 0.0f) continue;

    // 投影 MP 到 KF
    const float invz = 1 / p3Dc.at<float>(2);
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;
    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    // 投影点应在图像边界内
    if (!pKF->IsInImage(u, v)) continue;

    const float ur = u - bf * invz;  // 投影右点坐标

    const float maxDistance =
        pMP->GetMaxDistanceInvariance();  // 最大尺度无关距离*1.2
    const float minDistance =
        pMP->GetMinDistanceInvariance();  // 最小尺度无关距离*0.8
    cv::Mat PO = p3Dw - Ow;               //
    const float dist3D = cv::norm(PO);    // 距离

    // Depth must be inside the scale pyramid of the image
    // 判断是否位于尺度无关距离
    if (dist3D < minDistance || dist3D > maxDistance) continue;

    // Viewing angle must be less than 60 deg
    // 观测角必须小于 60 度
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D) continue;

    int nPredictedLevel = pMP->PredictScale(dist3D, pKF);  // 预测尺度

    // Search in a radius
    const float radius =
        th * pKF->mvScaleFactors[nPredictedLevel];  // 由预测尺度确定窗口半径

    const vector<size_t> vIndices =
        pKF->GetFeaturesInArea(u, v, radius);  // 投影点窗口范围内的候选关键点

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

    int bestDist = 256;
    int bestIdx = -1;
    // 遍历候选关键点  记录地图点的最佳匹配关键点
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;  // 关键点 id

      const cv::KeyPoint &kp = pKF->mvKeysUn[idx];  // 关键点

      const int &kpLevel = kp.octave;  // 关键点尺度级别

      // 如果层数不满足要求，忽略   特征点只能变远一个级别
      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel) continue;

      // 检查重投影误差，如果误差过大，说明不是匹配点；如果误差比较小，进一步进行描述子匹配
      // 如果关键点右点有效，检查重双目投影误差
      if (pKF->mvuRight[idx] >= 0) {
        // Check reprojection error in stereo
        const float &kpx = kp.pt.x;
        const float &kpy = kp.pt.y;
        const float &kpr = pKF->mvuRight[idx];
        const float ex = u - kpx;                      // x err
        const float ey = v - kpy;                      // y err
        const float er = ur - kpr;                     // ur err
        const float e2 = ex * ex + ey * ey + er * er;  // (total err)^2
        // 距离过大，说明不是匹配点
        if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8) continue;
      }
      // 如果关键点右点无效，检查重投影误差
      else {
        const float &kpx = kp.pt.x;
        const float &kpy = kp.pt.y;
        const float ex = u - kpx;
        const float ey = v - kpy;
        const float e2 = ex * ex + ey * ey;
        // 距离过大，说明不是匹配点
        if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99) continue;
      }

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);  // KP 描述子

      const int dist = DescriptorDistance(dMP, dKF);  // 汉明距离

      // 记录最佳匹配
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    // 如果已经有地图点观测，进行地图点融合  如果还没有观测，则添加新点观测
    // 最佳匹配小于低阈值
    if (bestDist <= TH_LOW) {
      MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);  // 旧地图点
      // 若观测存在，进行融合
      if (pMPinKF) {
        if (!pMPinKF->isBad()) {
          // 用观测数多的地图点继承少的那个，观测少的地图点被融合后置为坏点
          if (pMPinKF->Observations() > pMP->Observations())
            pMP->Replace(pMPinKF);
          else
            pMPinKF->Replace(pMP);
        }
      }
      // 观测不存在，添加新点的观测
      else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

/// @brief 回环地图点融合  给定相似变换，匹配地图点与关键帧
/// 添加新的观测关系，将需要融合的地图点放入向量，没有进行实际地图点融合
///     LoopClosing::SearchAndFuse
/// @param pKF 关键帧
/// @param Scw 相似变换
/// @param vpPoints 新地图点集
/// @param th 搜索窗口范围倍数
/// @param vpReplacePoint 需要替换的地图点  KF 观测的 MP，按照新 MP 顺序索引
/// @return 添加、融合的地图点数量
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw,
                     const vector<MapPoint *> &vpPoints, float th,
                     vector<MapPoint *> &vpReplacePoint) {
  // Get Calibration Parameters for later projection
  // 参数
  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;

  // Decompose Scw
  cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);      // sRcw
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));  // swc
  cv::Mat Rcw = sRcw / scw;                              // Rcw
  cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;         // tcw
  cv::Mat Ow = -Rcw.t() * tcw;                           // twc

  // Set of MapPoints already found in the KeyFrame
  const set<MapPoint *> spAlreadyFound =
      pKF->GetMapPoints();  // 关键帧地图点观测集合

  int nFused = 0;  // 添加、融合的地图点数量

  const int nPoints = vpPoints.size();  // 地图点总数

  // For each candidate MapPoint project and match
  // 遍历地图点，寻找匹配关键点
  for (int iMP = 0; iMP < nPoints; iMP++) {
    MapPoint *pMP = vpPoints[iMP];  // 地图点

    // Discard Bad MapPoints and already found
    // 如果点为坏点，或已经在此关键帧内被匹配，跳过
    if (pMP->isBad() || spAlreadyFound.count(pMP)) continue;

    // Get 3D Coords.
    cv::Mat p3Dw = pMP->GetWorldPos();  // 地图点空间位置

    // Transform into Camera Coords.
    cv::Mat p3Dc = Rcw * p3Dw + tcw;  // MP 相对 KF 位置

    // Depth must be positive
    // 检查深度为正
    if (p3Dc.at<float>(2) < 0.0f) continue;

    // Project into Image
    // 投影 MP 到 KF
    const float invz = 1.0 / p3Dc.at<float>(2);
    const float x = p3Dc.at<float>(0) * invz;
    const float y = p3Dc.at<float>(1) * invz;
    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    // 投影点应在图像边界内
    if (!pKF->IsInImage(u, v)) continue;

    // Depth must be inside the scale pyramid of the image
    const float maxDistance =
        pMP->GetMaxDistanceInvariance();  // 最大尺度无关距离*1.2
    const float minDistance =
        pMP->GetMinDistanceInvariance();  // 最小尺度无关距离*0.
    cv::Mat PO = p3Dw - Ow;
    const float dist3D = cv::norm(PO);  // 距离

    // 判断是否位于尺度无关距离
    if (dist3D < minDistance || dist3D > maxDistance) continue;

    // Viewing angle must be less than 60 deg
    // 观测角必须小于 60 度
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D) continue;

    // Compute predicted scale level
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);  // 预测尺度

    // Search in a radius
    const float radius =
        th * pKF->mvScaleFactors[nPredictedLevel];  // 由预测尺度确定窗口半径

    const vector<size_t> vIndices =
        pKF->GetFeaturesInArea(u, v, radius);  // 投影点窗口范围内的候选关键点

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

    int bestDist = INT_MAX;
    int bestIdx = -1;
    // 遍历候选关键点  记录地图点的最佳匹配关键点
    for (vector<size_t>::const_iterator vit = vIndices.begin();
         vit != vIndices.end(); vit++) {
      const size_t idx = *vit;                         // 关键点 id
      const int &kpLevel = pKF->mvKeysUn[idx].octave;  // 关键点

      // 如果层数不满足要求，忽略   特征点只能变远一个级别
      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel) continue;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);  // KP 描述子

      int dist = DescriptorDistance(dMP, dKF);  // 汉明距离

      // 记录最佳匹配
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    // 如果已经有地图点观测，将旧地图点放入替换向量
    // 如果还没有观测，则添加新点观测 最佳匹配小于低阈值
    if (bestDist <= TH_LOW) {
      MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);  // 旧地图点
      // 若观测存在
      if (pMPinKF) {
        // 如果已经有地图点观测，将旧地图点放入替换向量
        if (!pMPinKF->isBad()) vpReplacePoint[iMP] = pMPinKF;
      }
      // 观测不存在，添加新点的观测
      else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

/////////////////////////////////////////////////////////////////////////////////
///工具函数

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

/// @brief 计算 ORB 描述子距离汉明距离  快速算法
/// @param a 描述子
/// @param b 描述子
/// @return
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
  const int *pa = a.ptr<int32_t>();  // 32 bit 描述子 a
  const int *pb = b.ptr<int32_t>();  // 32 bit 描述子 b

  int dist = 0;  // 汉明距离

  // 遍历描述子每一部分   共 32*8 位描述子
  for (int i = 0; i < 8; i++, pa++, pb++) {
    unsigned int v = *pa ^ *pb;  // 按位异或，即两个 8 位描述子的汉明距离
    // 统计 32 位 int 中 1 个数的快速算法
    v = v - ((v >> 1) &
             0x55555555);  // 0x55555555 = 01010101 01010101 01010101 01010101
    v = (v & 0x33333333) +
        ((v >> 2) &
         0x33333333);  // 0x33333333 = 00110011 00110011 00110011 00110011
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >>
            24;  // 0x0F0F0F0F = 00001111 00001111 00001111 00001111
  }

  return dist;
}

/// @brief 寻找直方图中三个最大值的标号
/// @param histo 直方图
/// @param L 直方图宽度
/// @param ind1 最大值标号
/// @param ind2 第二大值标号
/// @param ind3 第三大值标号
void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1,
                                    int &ind2, int &ind3) {
  int max1 = 0;
  int max2 = 0;
  int max3 = 0;

  // 遍历直方图
  for (int i = 0; i < L; i++) {
    const int s = histo[i].size();  // 第 i 列特征点数
    if (s > max1) {
      // 如果大于最大
      max3 = max2;
      max2 = max1;
      max1 = s;
      ind3 = ind2;
      ind2 = ind1;
      ind1 = i;
    } else if (s > max2) {
      // 如果大于第二大
      max3 = max2;
      max2 = s;
      ind3 = ind2;
      ind2 = i;
    } else if (s > max3) {
      // 如果大于第三大
      max3 = s;
      ind3 = i;
    }
  }

  // 如果第二、第三的列远小于第一，抛弃他们
  if (max2 < 0.1f * (float)max1) {
    ind2 = -1;
    ind3 = -1;
  } else if (max3 < 0.1f * (float)max1) {
    ind3 = -1;
  }
}

/// @brief 依据视角余弦确定搜索半径   接近正视时半径小
/// @param viewCos
/// @return
float ORBmatcher::RadiusByViewingCos(const float &viewCos) {
  if (viewCos > 0.998)
    return 2.5;  // 小半径
  else
    return 4.0;  // 大半径
}

/// @brief 特征匹配极线距离检查   检查 KP2 与 KP1 按照基础矩阵 F12 投影 到 KF2
/// 图像平面极线的像素距离是否小于误差阈值
/// @param kp1 关键帧 1 关键点
/// @param kp2 关键帧 2 关键点
/// @param F12 基本矩阵
/// @param pKF2 关键帧 2
/// @return
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,
                                       const cv::KeyPoint &kp2,
                                       const cv::Mat &F12,
                                       const KeyFrame *pKF2) {
  // Epipolar line in second image l = x1'F12 = [a b c]
  // 对极约束为 x1'*F12*x2 = 0   x1 x2 为像素坐标
  // 其中 x1 在 KF2 图像平面上极线公式为 (x1'*F12)*x=0
  // 按照这个公式，可以计算 KP2 距离极线的距离
  // 如果距离过大，说明这两个特征点不满足对极约束，他们大概率来自不同地图点，或者提取自动态目标，需要剔除

  // 计算极线参数
  const float a = kp1.pt.x * F12.at<float>(0, 0) +
                  kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
  const float b = kp1.pt.x * F12.at<float>(0, 1) +
                  kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
  const float c = kp1.pt.x * F12.at<float>(0, 2) +
                  kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

  const float num =
      a * kp2.pt.x + b * kp2.pt.y + c;  // 如果 KP2 位于极线上，则等于 0

  const float den = a * a + b * b;  // 归一化系数的平方

  // 极线计算错误
  if (den == 0) return false;

  const float dsqr = num * num / den;  // 平方归一化误差

  return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];  // 判断是否小于阈值
}

}  // namespace ORB_SLAM2
