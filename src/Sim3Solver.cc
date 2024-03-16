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

#include "Sim3Solver.h"

#include <opencv2/imgproc/types_c.h>

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"
#include "ORBmatcher.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2 {

/// @brief 构造函数
/// @param pKF1 KF1
/// @param pKF2 KF2
/// @param vpMatched12 KF2 中地图点匹配  KF1 MP 到 KF2 MP  按照 KF1 KP 索引
/// @param bFixScale 固定尺度标志位
Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2,
                       const vector<MapPoint *> &vpMatched12,
                       const bool bFixScale)
    : mbFixScale(bFixScale), mnIterations(0), mnBestInliers(0) {
  mpKF1 = pKF1;
  mpKF2 = pKF2;

  vector<MapPoint *> vpKeyFrameMP1 =
      pKF1->GetMapPointMatches();  // KF1 MP 观测集

  mN1 = vpMatched12.size();

  mvpMapPoints1.reserve(mN1);
  mvpMapPoints2.reserve(mN1);
  mvpMatches12 = vpMatched12;
  mvnIndices1.reserve(mN1);
  mvX3Dc1.reserve(mN1);
  mvX3Dc2.reserve(mN1);

  cv::Mat Rcw1 = pKF1->GetRotation();
  cv::Mat tcw1 = pKF1->GetTranslation();
  cv::Mat Rcw2 = pKF2->GetRotation();
  cv::Mat tcw2 = pKF2->GetTranslation();

  mvAllIndices.reserve(mN1);

  size_t idx = 0;
  // 遍历输入匹配 MP 集，寻找其中有效匹配，填充信息
  for (int i1 = 0; i1 < mN1; i1++) {
    // 如果匹配存在
    if (vpMatched12[i1]) {
      MapPoint *pMP1 = vpKeyFrameMP1[i1];  // KF1 MP
      MapPoint *pMP2 = vpMatched12[i1];    // KF2 MP

      // 如果 KF1
      if (!pMP1) continue;

      if (pMP1->isBad() || pMP2->isBad()) continue;

      int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);  // KF1 KP id
      int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);  // KF2 KP id

      // 如果观测不存在。这个情况不应该出现
      if (indexKF1 < 0 || indexKF2 < 0) continue;

      const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];  // KF1 KP
      const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];  // KF2 KP

      const float sigmaSquare1 =
          pKF1->mvLevelSigma2[kp1.octave];  // KF1 KP 尺度平方
      const float sigmaSquare2 =
          pKF2->mvLevelSigma2[kp2.octave];  // KF2 KP 尺度平方

      // 按照尺度确定误差阈值
      mvnMaxError1.push_back(9.210 * sigmaSquare1);
      mvnMaxError2.push_back(9.210 * sigmaSquare2);

      mvpMapPoints1.push_back(pMP1);
      mvpMapPoints2.push_back(pMP2);
      mvnIndices1.push_back(i1);

      cv::Mat X3D1w = pMP1->GetWorldPos();     // KF1 MP tw
      mvX3Dc1.push_back(Rcw1 * X3D1w + tcw1);  // KF1 MP tc1

      cv::Mat X3D2w = pMP2->GetWorldPos();     // KF2 MP tw
      mvX3Dc2.push_back(Rcw2 * X3D2w + tcw2);  // KF2 MP tc2

      mvAllIndices.push_back(idx);
      idx++;
    }
  }

  mK1 = pKF1->mK;
  mK2 = pKF2->mK;

  // 计算 MP 投影像素坐标
  FromCameraToImage(mvX3Dc1, mvP1im1, mK1);
  FromCameraToImage(mvX3Dc2, mvP2im2, mK2);

  SetRansacParameters();  // RANSAC 使用默认参数
}

/// @brief 计算相机系地图点投影像素坐标
/// @param vP3Dc 地图点集 tc
/// @param vP2D 像素坐标集 (u,v)  输出
/// @param K 相机内参
void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc,
                                   vector<cv::Mat> &vP2D, cv::Mat K) {
  const float &fx = K.at<float>(0, 0);
  const float &fy = K.at<float>(1, 1);
  const float &cx = K.at<float>(0, 2);
  const float &cy = K.at<float>(1, 2);

  vP2D.clear();
  vP2D.reserve(vP3Dc.size());

  // 遍历地图点，计算投影
  for (size_t i = 0, iend = vP3Dc.size(); i < iend; i++) {
    const float invz = 1 / (vP3Dc[i].at<float>(2));
    const float x = vP3Dc[i].at<float>(0) * invz;
    const float y = vP3Dc[i].at<float>(1) * invz;

    vP2D.push_back((cv::Mat_<float>(2, 1) << fx * x + cx, fy * y + cy));
  }
}

/////////////////////////////////////////////////////////////////////////////////////
///RANSAC

/// @brief 设置 RANSAC 参数
/// @param probability 概率 (0.99)
/// @param minInliers 最少内点阈值 (6)
/// @param maxIterations 最大迭代次数 (300)
void Sim3Solver::SetRansacParameters(double probability, int minInliers,
                                     int maxIterations) {
  mRansacProb = probability;
  mRansacMinInliers = minInliers;
  mRansacMaxIts = maxIterations;

  // 有效匹配数量
  N = mvpMapPoints1.size();  // number of correspondences

  mvbInliersi.resize(N);

  // Adjust Parameters according to number of correspondences
  // 计算 RANSAC 迭代次数
  float epsilon = (float)mRansacMinInliers / N;

  // Set RANSAC iterations according to probability, epsilon, and max iterations
  int nIterations;

  if (mRansacMinInliers == N)
    nIterations = 1;
  else
    nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

  mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

  mnIterations = 0;
}

/// @brief 求解
/// @param vbInliers12 内点标志位 输出
/// @param nInliers 内点总数 输出
/// @return S12
cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers) {
  bool bFlag;
  return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers);
}

/// @brief 迭代
/// @param nIterations 本次最大迭代次数
/// @param bNoMore 不需要继续计算 输出
/// @param vbInliers 内点标志位 输出
/// @param nInliers 内点总数 输出
/// @return S12
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore,
                            vector<bool> &vbInliers, int &nInliers) {
  bNoMore = false;
  vbInliers = vector<bool>(mN1, false);
  nInliers = 0;

  if (N < mRansacMinInliers) {
    bNoMore = true;
    return cv::Mat();
  }

  vector<size_t> vAvailableIndices;  // 有效地图点序号

  cv::Mat P3Dc1i(3, 3, CV_32F);  // 3 个地图点 KF1 相机系位置
  cv::Mat P3Dc2i(3, 3, CV_32F);  // 3 个地图点 KF2 相机系位置

  int nCurrentIterations = 0;
  // 当迭代总数小于最大次数，且本次迭代总数小于本次迭代最大次数
  while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations) {
    nCurrentIterations++;
    mnIterations++;

    vAvailableIndices = mvAllIndices;

    // Get min set of points
    // 随机选取三对匹配    3 对不共线匹配足够确定位姿
    for (short i = 0; i < 3; ++i) {
      int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

      int idx = vAvailableIndices[randi];

      mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
      mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

      // 删除这一个匹配，防止重复选择
      vAvailableIndices[randi] = vAvailableIndices.back();
      vAvailableIndices.pop_back();
    }

    ComputeSim3(P3Dc1i, P3Dc2i);  // 求解

    CheckInliers();  // 内点检测

    // 如果内点数多于最佳采样下的内点数，记录
    if (mnInliersi >= mnBestInliers) {
      mvbBestInliers = mvbInliersi;
      mnBestInliers = mnInliersi;
      mBestT12 = mT12i.clone();
      mBestRotation = mR12i.clone();
      mBestTranslation = mt12i.clone();
      mBestScale = ms12i;

      // 只要达到最小内点要求，就返回结果
      if (mnInliersi > mRansacMinInliers) {
        nInliers = mnInliersi;
        for (int i = 0; i < N; i++)
          if (mvbInliersi[i]) vbInliers[mvnIndices1[i]] = true;
        return mBestT12;
      }
    }
  }

  // 注意，没有 Refine
  // 应该只是为优化提供初值，不需要精确结果

  // 采样失败，返回单位阵作为优化初值
  if (mnIterations >= mRansacMaxIts) bNoMore = true;

  return cv::Mat();
}

/// @brief 重投影误差检测内点
void Sim3Solver::CheckInliers() {
  vector<cv::Mat> vP1im2;  // KF1 MP 按当前 T21 投影 KF2 像素坐标
  vector<cv::Mat> vP2im1;  // KF2 MP 按当前 T12 投影 KF1 像素坐标
  // 计算互投影
  Project(mvX3Dc2, vP2im1, mT12i, mK1);
  Project(mvX3Dc1, vP1im2, mT21i, mK2);

  mnInliersi = 0;

  // 遍历有效匹配，计算重投影误差
  for (size_t i = 0; i < mvP1im1.size(); i++) {
    cv::Mat dist1 =
        mvP1im1[i] - vP2im1[i];  // KF1 MP 与 KF2 MP 在 KF1 投影坐标误差
    cv::Mat dist2 =
        vP1im2[i] - mvP2im2[i];  // KF1 MP 与 KF2 MP 在 KF2 投影坐标误差

    // 重投影误差
    const float err1 = dist1.dot(dist1);
    const float err2 = dist2.dot(dist2);

    // 如果两个重投影误差均小于阈值，则认为是内点
    if (err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i]) {
      mvbInliersi[i] = true;
      mnInliersi++;
    } else
      mvbInliersi[i] = false;
  }
}

/// @brief 计算世界系下地图点按给定变换矩阵、内参投影的像素坐标
/// 这里的"世界系"可能是另一个 KF
/// @param vP3Dw 地图点 tw
/// @param vP2D 地图点像素坐标 (u,v)  输出
/// @param Tcw Tcw
/// @param K K
void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D,
                         cv::Mat Tcw, cv::Mat K) {
  cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
  cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
  const float &fx = K.at<float>(0, 0);
  const float &fy = K.at<float>(1, 1);
  const float &cx = K.at<float>(0, 2);
  const float &cy = K.at<float>(1, 2);

  vP2D.clear();
  vP2D.reserve(vP3Dw.size());

  // 遍历地图点，计算相机系坐标，投影到图像平面
  for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++) {
    cv::Mat P3Dc = Rcw * vP3Dw[i] + tcw;
    const float invz = 1 / (P3Dc.at<float>(2));
    const float x = P3Dc.at<float>(0) * invz;
    const float y = P3Dc.at<float>(1) * invz;

    vP2D.push_back((cv::Mat_<float>(2, 1) << fx * x + cx, fy * y + cy));
  }
}

/////////////////////////////////////////////////////////////////////////////////////
///Sim3

/// @brief Sim3 求解  理论上可以进行 n 点匹配，实际只传入 3 个点对
/// 结果存放于成员变量
/// @param P1 n 个空间点 KF1 相机系坐标  cv::Mat 3*3
/// @param P2 n 个空间点 KF2 相机系坐标  cv::Mat 3*3
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2) {
  // Custom implementation of:
  // Horn 1987, Closed-form solution of absolute orientataion using unit
  // quaternions

  // Step 1: Centroid and relative coordinates
  // 计算输入点中心坐标、去中心坐标

  cv::Mat Pr1(P1.size(), P1.type());  // KF1 三点去中心坐标  3*3  Relative
                                      // coordinates to centroid (set 1)
  cv::Mat Pr2(P2.size(), P2.type());  // KF2 三点去中心坐标  3*3  Relative
                                      // coordinates to centroid (set 2)
  cv::Mat O1(3, 1, Pr1.type());  // KF1 中心坐标  Centroid of P1
  cv::Mat O2(3, 1, Pr2.type());  // KF2 中心坐标  Centroid of P2

  ComputeCentroid(P1, Pr1, O1);
  ComputeCentroid(P2, Pr2, O2);

  // Step 2: Compute M matrix
  // M 矩阵，去中心坐标张量积求和，等价于下面个两个矩阵相乘

  cv::Mat M = Pr2 * Pr1.t();

  // Step 3: Compute N matrix
  // 由 M 计算 N

  double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;  // N 中元素

  cv::Mat N(4, 4, P1.type());  // N 矩阵  4*4 对称矩阵

  N11 = M.at<float>(0, 0) + M.at<float>(1, 1) + M.at<float>(2, 2);
  N12 = M.at<float>(1, 2) - M.at<float>(2, 1);
  N13 = M.at<float>(2, 0) - M.at<float>(0, 2);
  N14 = M.at<float>(0, 1) - M.at<float>(1, 0);
  N22 = M.at<float>(0, 0) - M.at<float>(1, 1) - M.at<float>(2, 2);
  N23 = M.at<float>(0, 1) + M.at<float>(1, 0);
  N24 = M.at<float>(2, 0) + M.at<float>(0, 2);
  N33 = -M.at<float>(0, 0) + M.at<float>(1, 1) - M.at<float>(2, 2);
  N34 = M.at<float>(1, 2) + M.at<float>(2, 1);
  N44 = -M.at<float>(0, 0) - M.at<float>(1, 1) + M.at<float>(2, 2);

  N = (cv::Mat_<float>(4, 4) << N11, N12, N13, N14, N12, N22, N23, N24, N13,
       N23, N33, N34, N14, N24, N34, N44);

  // Step 4: Eigenvector of the highest eigenvalue
  // 求解 N 矩阵最大特征值对应特征向量  即待求旋转四元数

  cv::Mat eval;  // 特征值  从大到小排列
  cv::Mat evec;  // 特征向量  第一行为所需特征向量

  cv::eigen(
      N, eval,
      evec);  // 特征值分解  evec[0] is the quaternion of the desired rotation

  cv::Mat vec(1, 3, evec.type());  // 旋转向量，角轴
  (evec.row(0).colRange(1, 4))
      .copyTo(vec);  // extract imaginary part of the quaternion (sin*axis)

  // Rotation angle. sin is the norm of the imaginary part, cos is the real part
  double ang = atan2(norm(vec), evec.at<float>(0, 0));  // 旋转角度

  vec = 2 * ang * vec /
        norm(vec);  // 角轴，注意四元数转角是实际转角的一半  Angle-axis
                    // representation. quaternion angle is the half

  mR12i.create(3, 3, P1.type());  // 旋转矩阵

  cv::Rodrigues(
      vec,
      mR12i);  // 罗德里格斯公式 computes the rotation matrix from angle-axis

  // Step 5: Rotate set 2

  cv::Mat P3 = mR12i * Pr2;  // 旋转 F2 去中心坐标

  // Step 6: Scale
  // 这里的尺度计算没有用论文中的对称尺度公式
  // s = D/Sl

  // 如果不固定尺度  单目情况
  if (!mbFixScale) {
    double nom = Pr1.dot(P3);  // D
    cv::Mat aux_P3(P3.size(), P3.type());
    aux_P3 = P3;  // 这一行多余
    cv::pow(P3, 2, aux_P3);
    double den = 0;

    for (int i = 0; i < aux_P3.rows; i++) {
      for (int j = 0; j < aux_P3.cols; j++) {
        den += aux_P3.at<float>(i, j);
      }
    }

    ms12i = nom / den;  // D/Sl
  }
  // 如果固定尺度  双目/深度情况
  else
    ms12i = 1.0f;

  // Step 7: Translation
  // 计算平移

  mt12i.create(1, 3, P1.type());
  mt12i = O1 - ms12i * mR12i * O2;

  // Step 8: Transformation
  // 计算变换矩阵

  // Step 8.1 T12
  mT12i = cv::Mat::eye(4, 4, P1.type());

  cv::Mat sR = ms12i * mR12i;

  sR.copyTo(mT12i.rowRange(0, 3).colRange(0, 3));
  mt12i.copyTo(mT12i.rowRange(0, 3).col(3));

  // Step 8.2 T21

  mT21i = cv::Mat::eye(4, 4, P1.type());

  cv::Mat sRinv = (1.0 / ms12i) * mR12i.t();

  sRinv.copyTo(mT21i.rowRange(0, 3).colRange(0, 3));
  cv::Mat tinv = -sRinv * mt12i;
  tinv.copyTo(mT21i.rowRange(0, 3).col(3));
}

/// @brief 计算去中心坐标
/// @param P 原始坐标  输入
/// @param Pr 去中心坐标  输出
/// @param C 中心坐标  输出
void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C) {
  cv::reduce(P, C, 1, CV_REDUCE_SUM);  // 求和
  C = C / P.cols;                      // 平均

  // 遍历原始点集，求去中心坐标
  for (int i = 0; i < P.cols; i++) {
    Pr.col(i) = P.col(i) - C;
  }
}

/////////////////////////////////////////////////////////////////////////////////////
///其他

/// @brief 获取估计姿态 R12
/// @return 旋转矩阵
cv::Mat Sim3Solver::GetEstimatedRotation() { return mBestRotation.clone(); }

/// @brief 获取估计位置 t12
/// @return v3
cv::Mat Sim3Solver::GetEstimatedTranslation() {
  return mBestTranslation.clone();
}

/// @brief 获取估计尺度 s12
/// @return
float Sim3Solver::GetEstimatedScale() { return mBestScale; }

}  // namespace ORB_SLAM2
