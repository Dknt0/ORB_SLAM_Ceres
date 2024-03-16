/**
 * This file is part of ORB-SLAM2.
 * This file is a modified version of EPnP
 * <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
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

/**
 * Copyright (c) 2009, V. Lepetit, EPFL
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the FreeBSD Project
 */

#include "PnPsolver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <vector>

#include "Thirdparty/DBoW2/DUtils/Random.h"

using namespace std;

namespace ORB_SLAM2 {

/// @brief 构造函数
/// @param F 帧
/// @param vpMapPointMatches 地图点匹配
PnPsolver::PnPsolver(const Frame &F,
                     const vector<MapPoint *> &vpMapPointMatches)
    : pws(0),
      us(0),
      alphas(0),
      pcs(0),
      maximum_number_of_correspondences(0),
      number_of_correspondences(0),
      N(0),
      mnInliersi(0),
      mnIterations(0),
      mnBestInliers(0) {
  mvpMapPointMatches = vpMapPointMatches;
  mvP2D.reserve(F.mvpMapPoints.size());
  mvSigma2.reserve(F.mvpMapPoints.size());
  mvP3Dw.reserve(F.mvpMapPoints.size());
  mvKeyPointIndices.reserve(F.mvpMapPoints.size());
  mvAllIndices.reserve(F.mvpMapPoints.size());

  int idx = 0;
  // 遍历地图点匹配集，寻找有效地图点
  for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
    MapPoint *pMP = vpMapPointMatches[i];  // 地图点

    // 如果地图点观测存在且不为坏点
    if (pMP) {
      if (!pMP->isBad()) {
        const cv::KeyPoint &kp = F.mvKeysUn[i];  // 关键点

        mvP2D.push_back(kp.pt);
        mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);

        cv::Mat Pos = pMP->GetWorldPos();
        mvP3Dw.push_back(
            cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));

        mvKeyPointIndices.push_back(i);
        mvAllIndices.push_back(idx);

        idx++;
      }
    }
  }

  // Set camera calibration parameters
  fu = F.fx;
  fv = F.fy;
  uc = F.cx;
  vc = F.cy;

  SetRansacParameters();  // RANCAC 使用默认参数
}

/// @brief 析构函数
PnPsolver::~PnPsolver() {
  // 清堆区内存
  delete[] pws;
  delete[] us;
  delete[] alphas;
  delete[] pcs;
}

/////////////////////////////////////////////////////////////////////////////////////
///RANSAC

/// @brief 设置 RANSAC 参数
/// @param probability 概率 (0.99)
/// @param minInliers 最少内点数 (8)
/// @param maxIterations 最大迭代次数 (300)
/// @param minSet 最小集合数量 (4)
/// @param epsilon 期望内点/总数比例 (0.4)
/// @param th2 误差阈值平方 (5.991)
void PnPsolver::SetRansacParameters(double probability, int minInliers,
                                    int maxIterations, int minSet,
                                    float epsilon, float th2) {
  mRansacProb = probability;
  mRansacMinInliers = minInliers;
  mRansacMaxIts = maxIterations;
  mRansacEpsilon = epsilon;
  mRansacMinSet = minSet;

  N = mvP2D.size();  // number of correspondences

  mvbInliersi.resize(N);

  // Adjust Parameters according to number of correspondences

  // 确定最小内点数
  int nMinInliers = N * mRansacEpsilon;
  if (nMinInliers < mRansacMinInliers) nMinInliers = mRansacMinInliers;
  if (nMinInliers < minSet) nMinInliers = minSet;
  mRansacMinInliers = nMinInliers;

  if (mRansacEpsilon < (float)mRansacMinInliers / N)
    mRansacEpsilon = (float)mRansacMinInliers / N;

  // Set RANSAC iterations according to probability, epsilon, and max iterations
  // 确定迭代次数
  int nIterations;

  if (mRansacMinInliers == N)
    nIterations = 1;
  else
    nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(mRansacEpsilon, 3)));

  mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

  mvMaxError.resize(mvSigma2.size());
  for (size_t i = 0; i < mvSigma2.size(); i++)
    mvMaxError[i] = mvSigma2[i] * th2;
}

/// @brief 求解
/// @param vbInliers 内点标志位 输出
/// @param nInliers 内点总数
/// @return Tcw
cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers) {
  bool bFlag;
  return iterate(mRansacMaxIts, bFlag, vbInliers, nInliers);
}

/// @brief 迭代
/// @param nIterations RANSAC 迭代次数
/// @param bNoMore 不需要继续计算标志位 输出
/// @param vbInliers 内点标志位 输出
/// @param nInliers 内点总数 输出
/// @return Tcw
cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore,
                           vector<bool> &vbInliers, int &nInliers) {
  bNoMore = false;
  vbInliers.clear();
  nInliers = 0;

  set_maximum_number_of_correspondences(
      mRansacMinSet);  // 设置最大相关数 分配堆区内存

  if (N < mRansacMinInliers) {
    bNoMore = true;
    return cv::Mat();
  }

  vector<size_t> vAvailableIndices;  // 有效地图点序号

  int nCurrentIterations = 0;
  // 当 RANCAC 迭代总数没有超过最大值，且这本此计算没有超过给定最大迭代次数
  while (mnIterations < mRansacMaxIts || nCurrentIterations < nIterations) {
    nCurrentIterations++;  // 当前迭代次数
    mnIterations++;        // 迭代总数

    /* 随机选取一组最小关联集合，求解一次 EPnP，统计内点 */

    reset_correspondences();  // 重置关联

    vAvailableIndices = mvAllIndices;

    // Get min set of points
    // 随机选取 4 对关联，添加到本次 EPnP
    // 计算中，初步计算一个位姿估计，用于筛选内点
    for (short i = 0; i < mRansacMinSet; ++i) {
      int randi = DUtils::Random::RandomInt(
          0, vAvailableIndices.size() - 1);  // 随机关联序号

      int idx = vAvailableIndices[randi];  // 有效地图点序号

      add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z,
                         mvP2D[idx].x, mvP2D[idx].y);  // 添加一组关联

      // 删除这此循环添加的关联，防止重复添加
      vAvailableIndices[randi] = vAvailableIndices.back();
      vAvailableIndices.pop_back();
    }

    // Compute camera pose
    // 计算相机位姿
    compute_pose(mRi, mti);

    // Check inliers
    // 检查内点
    CheckInliers();

    // 如果当前内点数多余 RANSAC 最小内点阈值，记录当前状态，并进行改善
    if (mnInliersi >= mRansacMinInliers) {
      // If it is the best solution so far, save it
      // 如果当前内点数多余 RANSAC 最小内点阈值
      if (mnInliersi > mnBestInliers) {
        mvbBestInliers = mvbInliersi;  // 记录内点标志位
        mnBestInliers = mnInliersi;    // 记录内点数量

        // 创建矩阵
        cv::Mat Rcw(3, 3, CV_64F, mRi);
        cv::Mat tcw(3, 1, CV_64F, mti);
        // 类型转换
        Rcw.convertTo(Rcw, CV_32F);
        tcw.convertTo(tcw, CV_32F);
        // 当前最佳位姿
        mBestTcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(mBestTcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(mBestTcw.rowRange(0, 3).col(3));
      }

      // 改善，保存改善结果
      if (Refine()) {
        nInliers = mnRefinedInliers;
        vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
        for (int i = 0; i < N; i++) {
          if (mvbRefinedInliers[i]) vbInliers[mvKeyPointIndices[i]] = true;
        }
        return mRefinedTcw.clone();
      }
    }
  }

  // 如果超过最大迭代次数
  if (mnIterations >= mRansacMaxIts) {
    bNoMore = true;  // 不进行更多计算
    // 如果最佳内点数多于内点阈值，说明结果有效
    if (mnBestInliers >= mRansacMinInliers) {
      nInliers = mnBestInliers;
      vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
      for (int i = 0; i < N; i++) {
        if (mvbBestInliers[i]) vbInliers[mvKeyPointIndices[i]] = true;
      }
      return mBestTcw.clone();
    }
  }

  return cv::Mat();
}

/// @brief 修正 使用当前最多内点集，计算 EPnP
/// @return
bool PnPsolver::Refine() {
  vector<int> vIndices;  // 内点序号集
  vIndices.reserve(mvbBestInliers.size());

  // 遍历筛选关联集中内点
  for (size_t i = 0; i < mvbBestInliers.size(); i++) {
    if (mvbBestInliers[i]) {
      vIndices.push_back(i);
    }
  }

  set_maximum_number_of_correspondences(vIndices.size());  // 重置关联数

  reset_correspondences();  // 重置关联

  // 添加关联
  for (size_t i = 0; i < vIndices.size(); i++) {
    int idx = vIndices[i];
    add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z,
                       mvP2D[idx].x, mvP2D[idx].y);
  }

  // Compute camera pose
  compute_pose(mRi, mti);  // 估计

  // Check inliers
  CheckInliers();  // 重新检查内点

  mnRefinedInliers = mnInliersi;
  mvbRefinedInliers = mvbInliersi;

  // 如果当前内点数多余 RANSAC 最小内点阈值，保留结果
  // 这里应该是会保留的
  if (mnInliersi > mRansacMinInliers) {
    cv::Mat Rcw(3, 3, CV_64F, mRi);
    cv::Mat tcw(3, 1, CV_64F, mti);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);
    mRefinedTcw = cv::Mat::eye(4, 4, CV_32F);
    Rcw.copyTo(mRefinedTcw.rowRange(0, 3).colRange(0, 3));
    tcw.copyTo(mRefinedTcw.rowRange(0, 3).col(3));
    return true;
  }

  return false;
}

/// @brief 重投影误差筛选内点
void PnPsolver::CheckInliers() {
  // 重置内点数量
  mnInliersi = 0;

  // 遍历所有有效关联，筛选内点
  for (int i = 0; i < N; i++) {
    cv::Point3f P3Dw = mvP3Dw[i];  // 地图点空间位置
    cv::Point2f P2D = mvP2D[i];    // 特征点像素位置

    // 依据当前 mRi mti 将空间点投影到图像平面
    float Xc =
        mRi[0][0] * P3Dw.x + mRi[0][1] * P3Dw.y + mRi[0][2] * P3Dw.z + mti[0];
    float Yc =
        mRi[1][0] * P3Dw.x + mRi[1][1] * P3Dw.y + mRi[1][2] * P3Dw.z + mti[1];
    float invZc = 1 / (mRi[2][0] * P3Dw.x + mRi[2][1] * P3Dw.y +
                       mRi[2][2] * P3Dw.z + mti[2]);

    double ue = uc + fu * Xc * invZc;
    double ve = vc + fv * Yc * invZc;

    float distX = P2D.x - ue;
    float distY = P2D.y - ve;

    float error2 = distX * distX + distY * distY;  // 重投影误差

    // 如果重投影误差小于阈值，认为是内点
    if (error2 < mvMaxError[i]) {
      mvbInliersi[i] = true;
      mnInliersi++;
    } else {
      mvbInliersi[i] = false;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////
///EPnP 源码

/// @brief EPnP+LHM 估计相机位姿 核心函数
/// @param R 旋转矩阵 Rcw 输出
/// @param t 位置 tcw 输出
/// @return 重投影误差
double PnPsolver::compute_pose(double R[3][3], double t[3]) {
  choose_control_points();            // 选择控制点
  compute_barycentric_coordinates();  // 计算空间点的控制点的权重 alphas

  CvMat *M =
      cvCreateMat(2 * number_of_correspondences, 12, CV_64F);  // M 矩阵 2n*12

  /* 从关联计算 M，计算奇异值向两 */

  // 遍历所有关联
  for (int i = 0; i < number_of_correspondences; i++)
    fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);  // 填充 M 矩阵

  double mtm[12 * 12], d[12], ut[12 * 12];
  CvMat MtM = cvMat(12, 12, CV_64F, mtm);  // M^T*M  12*12
  CvMat D = cvMat(12, 1, CV_64F, d);  // SVD 奇异值  透视相机1个0  仿射相机4个0
  CvMat Ut = cvMat(12, 12, CV_64F, ut);  // SVD 右奇异值向量 U^T

  cvMulTransposed(M, &MtM, 1);                            // 计算 M^T*M
  cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);  // SVD 分解
  cvReleaseMat(&M);                                       // 释放 M

  /* 构建非线性最小二乘，求解控制点的核向量系数 */
  // NLS 问题
  // 残差  e = LB - rho
  // beta0 对应最后一个核向量，即最小奇异值对应的核向量  beta1
  // 为倒数第二个，以此类推 B = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]^T B
  // 代表 beta 的二次形   B11 = beta1 * beta1

  // 四个控制点两两组合，C42=6 组残差  4个beta取其二，包括自组合，C42+4=10
  // 个估计量  所以L是6*10, rho为6维 具体优化过程依据 N
  // 不同，即选择的核向量个数不同，选择 L 的不同列计算不同的 beta 组合
  double l_6x10[6 * 10], rho[6];
  CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);  // L 6*10
  CvMat Rho = cvMat(6, 1, CV_64F, rho);         // Rho

  compute_L_6x10(ut, l_6x10);  // 最小二乘系数项
  compute_rho(rho);            // 最小二乘常数项

  // 下面的算法和论文中的不一样!!!!!!!!!!!!!!
  // 全部为 N=4 的情况，但初值的假设不同

  // 通过三种不同的近似方法为 beta 提供初值，然后分别使用牛顿高斯法优化，求解
  // beta，从三种结果中选取重投影误差最小的返回
  // 这里保留了计算四种结果的空间，实际只计算了三种
  double Betas[4][4];    // betas 核向量权重
  double rep_errors[4];  // 重投影误差
  double Rs[4][3][3];    // 姿态结果
  double ts[4][3];       // 位置结果

  find_betas_approx_1(&L_6x10, &Rho, Betas[1]);  // N = 4 计算 beta 初值
  gauss_newton(&L_6x10, &Rho, Betas[1]);         // 优化求解 beta
  rep_errors[1] =
      compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);  // 计算重投影误差

  find_betas_approx_2(&L_6x10, &Rho, Betas[2]);  // N = 2
  gauss_newton(&L_6x10, &Rho, Betas[2]);         // 优化求解 beta
  rep_errors[2] =
      compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);  // 计算重投影误差

  find_betas_approx_3(&L_6x10, &Rho, Betas[3]);  // N = 3
  gauss_newton(&L_6x10, &Rho, Betas[3]);         // 优化求解 beta
  rep_errors[3] =
      compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);  // 计算重投影误差

  // 忽略了一种方法

  // 寻找最小重投影误差
  int N = 1;
  if (rep_errors[2] < rep_errors[1]) N = 2;
  if (rep_errors[3] < rep_errors[N]) N = 3;

  copy_R_and_t(Rs[N], ts[N], R, t);  // 填充结果

  return rep_errors[N];  // 返回最小重投影误差
}

/// @brief 设置最大相关数 分配堆区内存
/// @param n
void PnPsolver::set_maximum_number_of_correspondences(int n) {
  if (maximum_number_of_correspondences < n) {
    // 清内存
    if (pws != 0) delete[] pws;
    if (us != 0) delete[] us;
    if (alphas != 0) delete[] alphas;
    if (pcs != 0) delete[] pcs;

    // 分内存
    maximum_number_of_correspondences = n;
    pws = new double[3 * maximum_number_of_correspondences];
    us = new double[2 * maximum_number_of_correspondences];
    alphas = new double[4 * maximum_number_of_correspondences];
    pcs = new double[3 * maximum_number_of_correspondences];
  }
}

/// @brief 重置观测数
/// @param
void PnPsolver::reset_correspondences(void) { number_of_correspondences = 0; }

/// @brief 添加一组 3D-2D 投影关联，当前关联数量 +1
/// @param X 空间点 x
/// @param Y 空间点 y
/// @param Z 空间点 z
/// @param u 像素点 u
/// @param v 像素点 v
void PnPsolver::add_correspondence(double X, double Y, double Z, double u,
                                   double v) {
  pws[3 * number_of_correspondences] = X;
  pws[3 * number_of_correspondences + 1] = Y;
  pws[3 * number_of_correspondences + 2] = Z;

  us[2 * number_of_correspondences] = u;
  us[2 * number_of_correspondences + 1] = v;

  number_of_correspondences++;
}

/// @brief 选择控制点
/// @param
void PnPsolver::choose_control_points(void) {
  // Take C0 as the reference points centroid:
  // C0 为空间点重心
  cws[0][0] = cws[0][1] = cws[0][2] = 0;
  for (int i = 0; i < number_of_correspondences; i++)
    for (int j = 0; j < 3; j++) cws[0][j] += pws[3 * i + j];

  for (int j = 0; j < 3; j++) cws[0][j] /= number_of_correspondences;

  // Take C1, C2, and C3 from PCA on the reference points:
  // C1, C2, C3 通过空间点主方向确定
  CvMat *PW0 =
      cvCreateMat(number_of_correspondences, 3, CV_64F);  // 空间点坐标 n*3

  double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
  CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);  // PW0^T * PW0
  CvMat DC = cvMat(3, 1, CV_64F, dc);            // D 奇异值
  CvMat UCt = cvMat(3, 3, CV_64F, uct);          // U^T 主方向

  // 填充空间点坐标
  for (int i = 0; i < number_of_correspondences; i++)
    for (int j = 0; j < 3; j++)
      PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];

  cvMulTransposed(PW0, &PW0tPW0, 1);  // 计算 PW0^T * PW0
  cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);  // SVD 分解

  cvReleaseMat(&PW0);  // 释放内存

  // 填充 C1, C2, C3 点坐标
  for (int i = 1; i < 4; i++) {
    double k = sqrt(dc[i - 1] / number_of_correspondences);  // 系数
    for (int j = 0; j < 3; j++)
      cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
  }
}

/// @brief 计算空间点的控制点的权重 alphas
/// @param
void PnPsolver::compute_barycentric_coordinates(void) {
  // 坐标变换
  double cc[3 * 3], cc_inv[3 * 3];
  CvMat CC = cvMat(3, 3, CV_64F, cc);  // 控制点坐标系在世界坐标系下的表示
  CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);  // 世界系在控制点系下的表示

  // cc 为控制点坐标系在世界坐标系下的表示
  for (int i = 0; i < 3; i++)
    for (int j = 1; j < 4; j++) cc[3 * i + j - 1] = cws[j][i] - cws[0][i];

  cvInvert(&CC, &CC_inv, CV_SVD);
  double *ci = cc_inv;
  // 遍历空间点
  for (int i = 0; i < number_of_correspondences; i++) {
    double *pi = pws + 3 * i;
    double *a = alphas + 4 * i;

    // 坐标变换
    // 空间点在控制点系下坐标       cod_c = CC_inv * (Pi - C0)
    // 通过控制点表达空间点坐标     a = |-1 -1 -1 | * cod_c + | 1 |
    //                              | 1  0  0 |           | 0 |
    //                              | 0  1  0 |           | 0 |
    //                              | 0  0  1 |           | 0 |
    // a[1:3] = CC_inv * (Pi - C0)
    // a[0] = 1 - a[1] - a[2] - a[3]
    for (int j = 0; j < 3; j++)
      a[1 + j] = ci[3 * j] * (pi[0] - cws[0][0]) +
                 ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                 ci[3 * j + 2] * (pi[2] - cws[0][2]);
    a[0] = 1.0f - a[1] - a[2] - a[3];  // 因为以 C0 为原点，所以 +1
  }
}

/// @brief 填充 M 矩阵  用一组关联填充 M 中的两行
/// @param M M 矩阵
/// @param row 起始行数
/// @param as alpha 系数  控制点权重
/// @param u 图像坐标
/// @param v 图像坐标
void PnPsolver::fill_M(CvMat *M, const int row, const double *as,
                       const double u, const double v) {
  // 一对关联可以提供 2 个约束，所以填充两行
  double *M1 = M->data.db + row * 12;  // 第一行
  double *M2 = M1 + 12;                // 第二行

  // 对应四个控制点位置
  for (int i = 0; i < 4; i++) {
    M1[3 * i] = as[i] * fu;            // cix 系数
    M1[3 * i + 1] = 0.0;               // ciy 系数
    M1[3 * i + 2] = as[i] * (uc - u);  // ciz 系数

    M2[3 * i] = 0.0;                   // cix 系数
    M2[3 * i + 1] = as[i] * fv;        // ciy 系数
    M2[3 * i + 2] = as[i] * (vc - v);  // ciz 系数
  }
}

/// @brief 计算控制点相机坐标
/// @param betas betas
/// @param ut SVD 右奇异值向量 U^T
void PnPsolver::compute_ccs(const double *betas, const double *ut) {
  // 控制点坐标初始化
  for (int i = 0; i < 4; i++) ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

  // 控制点坐标是最后四个 SVD 右奇异值向量的线性组合
  // 注意 beta0 对应最后一个核向量
  for (int i = 0; i < 4; i++) {
    const double *v = ut + 12 * (11 - i);
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++) ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

/// @brief 计算空间点相机坐标
/// @param
void PnPsolver::compute_pcs(void) {
  // 遍历空间点
  for (int i = 0; i < number_of_correspondences; i++) {
    double *a = alphas + 4 * i;
    double *pc = pcs + 3 * i;

    // 空间点像素坐标是控制点坐标的线性组合
    for (int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] +
              a[3] * ccs[3][j];
  }
}

/// @brief 拷贝姿态与位置
/// @param R_src 输入旋转矩阵
/// @param t_src 输入位置
/// @param R_dst 输出旋转矩阵
/// @param t_dst 输出位置
void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
                             double R_dst[3][3], double t_dst[3]) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

/// @brief 计算两三维向量差值模长的平方
/// @param p1
/// @param p2
/// @return
double PnPsolver::dist2(const double *p1, const double *p2) {
  return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) +
         (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

/// @brief 计算两三维向量点乘
/// @param v1
/// @param v2
/// @return
double PnPsolver::dot(const double *v1, const double *v2) {
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/// @brief 重投影误差
/// @param R 姿态
/// @param t 位置
/// @return
double PnPsolver::reprojection_error(const double R[3][3], const double t[3]) {
  double sum2 = 0.0;  // 重投影误差

  // 遍历，计算每一组关联的误差
  for (int i = 0; i < number_of_correspondences; i++) {
    double *pw = pws + 3 * i;                      // 空间点坐标
    double Xc = dot(R[0], pw) + t[0];              // 相机系 x
    double Yc = dot(R[1], pw) + t[1];              // 相机系 y
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);  // 相机系逆深度
    double ue = uc + fu * Xc * inv_Zc;             // 图像重投影 u
    double ve = vc + fv * Yc * inv_Zc;             // 图像重投影 v
    double u = us[2 * i], v = us[2 * i + 1];       // 真实 u, v

    sum2 += sqrt((u - ue) * (u - ue) +
                 (v - ve) * (v - ve));  // 一组匹配的重投影误差
  }

  return sum2 / number_of_correspondences;  // 平均误差
}

/// @brief ICP 估计位姿 SVD 分解法
/// @param R
/// @param t
void PnPsolver::estimate_R_and_t(double R[3][3], double t[3]) {
  double pc0[3];  // 相机系下空间点平均坐标
  double pw0[3];  // 世界系下空间点平均坐标

  // 初始化平均坐标
  pc0[0] = pc0[1] = pc0[2] = 0.0;
  pw0[0] = pw0[1] = pw0[2] = 0.0;

  // 计算平均坐标
  for (int i = 0; i < number_of_correspondences; i++) {
    const double *pc = pcs + 3 * i;
    const double *pw = pws + 3 * i;

    for (int j = 0; j < 3; j++) {
      pc0[j] += pc[j];
      pw0[j] += pw[j];
    }
  }
  for (int j = 0; j < 3; j++) {
    pc0[j] /= number_of_correspondences;
    pw0[j] /= number_of_correspondences;
  }

  // SVD 分解求姿态
  // ABt = UDV'
  double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
  CvMat ABt = cvMat(3, 3, CV_64F, abt);
  CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
  CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
  CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

  // 计算相机系去中心坐标与世界系去中心坐标张量积的和 ABt
  // 通常，这个矩阵记为 W
  cvSetZero(&ABt);
  for (int i = 0; i < number_of_correspondences; i++) {
    double *pc = pcs + 3 * i;
    double *pw = pws + 3 * i;

    for (int j = 0; j < 3; j++) {
      abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
      abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
      abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
    }
  }

  // 计算 SVD 分解
  cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

  // R = UV'
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

  // 行列式 det(R)
  const double det = R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] +
                     R[0][2] * R[1][0] * R[2][1] - R[0][2] * R[1][1] * R[2][0] -
                     R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

  // 如果行列式小于 0，需要将 R 变回 SO(3)
  if (det < 0) {
    R[2][0] = -R[2][0];
    R[2][1] = -R[2][1];
    R[2][2] = -R[2][2];
  }

  // t = pc0 - Rpw0
  t[0] = pc0[0] - dot(R[0], pw0);
  t[1] = pc0[1] - dot(R[1], pw0);
  t[2] = pc0[2] - dot(R[2], pw0);
}

/// @brief 输出位姿到 cout
/// @param R 旋转矩阵
/// @param t 位置
void PnPsolver::print_pose(const double R[3][3], const double t[3]) {
  cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
  cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
  cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

/// @brief 求解符号，保证空间点位于相机前方
/// @param
void PnPsolver::solve_for_sign(void) {
  // 如果第一个点的 z 为负，说明点位于相机后方
  if (pcs[2] < 0.0) {
    // 对控制点坐标取反
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 3; j++) ccs[i][j] = -ccs[i][j];

    // 对空间点坐标取反
    for (int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

/// @brief 计算位姿与重投影误差
/// @param ut SVD 右奇异值向量 U^T
/// @param betas beta
/// @param R 姿态 输出
/// @param t 位置 输出
/// @return 重投影误差
double PnPsolver::compute_R_and_t(const double *ut, const double *betas,
                                  double R[3][3], double t[3]) {
  compute_ccs(betas, ut);  // 计算控制点相机坐标
  compute_pcs();           // 计算空间点相机坐标

  solve_for_sign();  // 保证空间点位于相机前方

  // 当相机坐标系下空间点、控制点坐标已知时，位姿估计问题转化为 ICP，可以通过
  // SVD 分解求得解析解

  estimate_R_and_t(R, t);  // 估计姿态和位置

  return reprojection_error(R, t);  // 计算重投影误差
}

/// @brief beta 计算  N = 4
/// @param L_6x10 最小二乘系数项
/// @param Rho 最小二乘常数项
/// @param betas beta 结果
void PnPsolver::find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho,
                                    double *betas) {
  // 注意这里没有用到全部信息，和论文里不一样
  // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_1 = [B11 B12     B13         B14]

  double l_6x4[6 * 4];  // 对应上面的 betas_approx_1
  double b4[4];         // beta
  CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
  CvMat B4 = cvMat(4, 1, CV_64F, b4);

  // 填充 L_6x4
  for (int i = 0; i < 6; i++) {
    cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
  }

  // 求解线性方程组
  cvSolve(&L_6x4, Rho, &B4, CV_SVD);

  // 保证空间点位于相机前方
  if (b4[0] < 0) {
    betas[0] = sqrt(-b4[0]);
    betas[1] = -b4[1] / betas[0];
    betas[2] = -b4[2] / betas[0];
    betas[3] = -b4[3] / betas[0];
  } else {
    betas[0] = sqrt(b4[0]);
    betas[1] = b4[1] / betas[0];
    betas[2] = b4[2] / betas[0];
    betas[3] = b4[3] / betas[0];
  }
}

/// @brief beta 计算  N = 2
/// @param L_6x10
/// @param Rho
/// @param betas
void PnPsolver::find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho,
                                    double *betas) {
  // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_2 = [B11 B12 B22                            ]

  double l_6x3[6 * 3];  // betas_approx_2
  double b3[3];         // beta
  CvMat L_6x3 = cvMat(6, 3, CV_64F, l_6x3);
  CvMat B3 = cvMat(3, 1, CV_64F, b3);

  // 填充 L_6x3
  for (int i = 0; i < 6; i++) {
    cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
  }

  // 求解线性方程组
  cvSolve(&L_6x3, Rho, &B3, CV_SVD);

  // 保证空间点位于相机前方
  if (b3[0] < 0) {
    betas[0] = sqrt(-b3[0]);
    betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
  } else {
    betas[0] = sqrt(b3[0]);
    betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
  }

  if (b3[1] < 0) betas[0] = -betas[0];

  // 剩余 beta 赋 0
  betas[2] = 0.0;
  betas[3] = 0.0;
}

/// @brief beta 计算  N = 3
/// @param L_6x10
/// @param Rho
/// @param betas
void PnPsolver::find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho,
                                    double *betas) {
  // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_3 = [B11 B12 B22 B13 B23                    ]

  double l_6x5[6 * 5];  // 对应上面的 betas_approx_3
  double b5[5];         // beta
  CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
  CvMat B5 = cvMat(5, 1, CV_64F, b5);

  // 填充 L_6x5
  for (int i = 0; i < 6; i++) {
    cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
    cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
  }

  // 求解线性方程组
  cvSolve(&L_6x5, Rho, &B5, CV_SVD);

  // 保证空间点位于相机前方
  if (b5[0] < 0) {
    betas[0] = sqrt(-b5[0]);
    betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
  } else {
    betas[0] = sqrt(b5[0]);
    betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
  }
  if (b5[1] < 0) betas[0] = -betas[0];
  betas[2] = b5[3] / betas[0];
  betas[3] = 0.0;  // beta4 赋 0
}

/// @brief 计算最小二乘系数项
/// @param ut SVD 右奇异值向量 U^T
/// @param l_6x10 最小二乘系数项  输出
void PnPsolver::compute_L_6x10(const double *ut, double *l_6x10) {
  const double *v[4];  // 核向量

  // 核向量长度为 12，对应四个 控制点坐标
  v[0] = ut + 12 * 11;  // 最小奇异值 核向量
  v[1] = ut + 12 * 10;  // 次小奇异值 核向量
  v[2] = ut + 12 * 9;   // 第三小奇异值 核向量
  v[3] = ut + 12 * 8;   // 第四小奇异值 核向量

  double dv[4][6][3];  // 核向量之间的差值

  // 遍历，对于每一个控制点坐标，分别计算核向量差值
  for (int i = 0; i < 4; i++) {
    int a = 0, b = 1;
    // C42 = 6 种差值，每个差向量维数为 3
    for (int j = 0; j < 6; j++) {
      dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

      b++;
      if (b > 3) {
        a++;
        b = a + 1;
      }
    }
  }

  // 计算 l_6x10 第 i 行 betaij 的系数
  for (int i = 0; i < 6; i++) {
    double *row = l_6x10 + 10 * i;

    row[0] = dot(dv[0][i], dv[0][i]);
    row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
    row[2] = dot(dv[1][i], dv[1][i]);
    row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
    row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
    row[5] = dot(dv[2][i], dv[2][i]);
    row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
    row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
    row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
    row[9] = dot(dv[3][i], dv[3][i]);
  }
}

/// @brief 计算最小二乘常数项
/// @param rho 常数项 返回值
void PnPsolver::compute_rho(double *rho) {
  // 共 6 个残差项
  // 每个残差项是对应控制点在世界坐标系下的距离
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

/// @brief 高斯牛顿一轮迭代 计算 A b
/// @param l_6x10 L 输入
/// @param rho rho 输入
/// @param betas 初值 输入
/// @param A 雅可比矩阵 输出
/// @param b 负残差 输出
void PnPsolver::compute_A_and_b_gauss_newton(const double *l_6x10,
                                             const double *rho, double betas[4],
                                             CvMat *A, CvMat *b) {
  for (int i = 0; i < 6; i++) {
    const double *rowL = l_6x10 + i * 10;  // L 的一行
    double *rowA = A->data.db + i * 4;

    // 计算线性化雅可比
    rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] +
              rowL[6] * betas[3];
    rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] +
              rowL[7] * betas[3];
    rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +
              rowL[8] * betas[3];
    rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] +
              2 * rowL[9] * betas[3];

    // 计算负残差
    cvmSet(b, i, 0,
           rho[i] -
               (rowL[0] * betas[0] * betas[0] + rowL[1] * betas[0] * betas[1] +
                rowL[2] * betas[1] * betas[1] + rowL[3] * betas[0] * betas[2] +
                rowL[4] * betas[1] * betas[2] + rowL[5] * betas[2] * betas[2] +
                rowL[6] * betas[0] * betas[3] + rowL[7] * betas[1] * betas[3] +
                rowL[8] * betas[2] * betas[3] + rowL[9] * betas[3] * betas[3]));
  }
}

/// @brief 高斯牛顿
/// @param L_6x10 最小二乘系数
/// @param Rho 最小二乘常数
/// @param betas beta 初值，输出
void PnPsolver::gauss_newton(const CvMat *L_6x10, const CvMat *Rho,
                             double betas[4]) {
  const int iterations_number = 5;

  double a[6 * 4], b[6], x[4];
  CvMat A = cvMat(6, 4, CV_64F, a);  // 雅可比
  CvMat B = cvMat(6, 1, CV_64F, b);  // 误差
  CvMat X = cvMat(4, 1, CV_64F, x);  // 增量

  // 迭代 5 次
  for (int k = 0; k < iterations_number; k++) {
    compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db, betas, &A,
                                 &B);  // 高斯牛顿一轮迭代 计算 A b
    // 注意这个 b 是负残差
    // delta_beta = (A^T*A)^(-1)*A^T*b
    qr_solve(&A, &B, &X);  // QR 分解求增量

    for (int i = 0; i < 4; i++) betas[i] += x[i];  // 增量更新
  }
}

/// @brief QR 分解 计算X  X=(AT*A)^(-1)*AT*b  存疑，变量命名、矩阵运算太乱了
/// @param A 雅可比
/// @param b 残差
/// @param X beta 增量  输出
void PnPsolver::qr_solve(CvMat *A, CvMat *b, CvMat *X) {
  static int max_nr = 0;
  static double *A1;
  static double *A2;

  const int nr = A->rows;  // 雅可比行数
  const int nc = A->cols;  // 雅可比列数

  // 重分配内存
  if (max_nr != 0 && max_nr < nr) {
    delete[] A1;
    delete[] A2;
  }
  if (max_nr < nr) {
    max_nr = nr;
    A1 = new double[nr];
    A2 = new double[nr];
  }

  double *pA = A->data.db;
  double *ppAkk = pA;
  // 遍历列
  for (int k = 0; k < nc; k++) {
    double *ppAik = ppAkk, eta = fabs(*ppAik);
    for (int i = k + 1; i < nr; i++) {
      double elt = fabs(*ppAik);
      if (eta < elt) eta = elt;
      ppAik += nc;
    }

    if (eta == 0) {
      A1[k] = A2[k] = 0.0;
      // 代码中口吐芬芳，学到了
      cerr << "God damnit, A is singular, this shouldn't happen." << endl;
      return;
    } else {
      double *ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
      for (int i = k; i < nr; i++) {
        *ppAik *= inv_eta;
        sum += *ppAik * *ppAik;
        ppAik += nc;
      }
      double sigma = sqrt(sum);
      if (*ppAkk < 0) sigma = -sigma;
      *ppAkk += sigma;
      A1[k] = sigma * *ppAkk;
      A2[k] = -eta * sigma;
      for (int j = k + 1; j < nc; j++) {
        double *ppAik = ppAkk, sum = 0;
        for (int i = k; i < nr; i++) {
          sum += *ppAik * ppAik[j - k];
          ppAik += nc;
        }
        double tau = sum / A1[k];
        ppAik = ppAkk;
        for (int i = k; i < nr; i++) {
          ppAik[j - k] -= tau * *ppAik;
          ppAik += nc;
        }
      }
    }
    ppAkk += nc + 1;
  }

  // b <- Qt b
  double *ppAjj = pA, *pb = b->data.db;
  for (int j = 0; j < nc; j++) {
    double *ppAij = ppAjj, tau = 0;
    for (int i = j; i < nr; i++) {
      tau += *ppAij * pb[i];
      ppAij += nc;
    }
    tau /= A1[j];
    ppAij = ppAjj;
    for (int i = j; i < nr; i++) {
      pb[i] -= tau * *ppAij;
      ppAij += nc;
    }
    ppAjj += nc + 1;
  }

  // X = R-1 b
  double *pX = X->data.db;
  pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
  for (int i = nc - 2; i >= 0; i--) {
    double *ppAij = pA + i * nc + (i + 1), sum = 0;

    for (int j = i + 1; j < nc; j++) {
      sum += *ppAij * pX[j];
      ppAij++;
    }
    pX[i] = (pb[i] - sum) / A2[i];
  }
}

/// @brief 相关误差  貌似没有用到
/// @param rot_err
/// @param transl_err
/// @param Rtrue
/// @param ttrue
/// @param Rest
/// @param test
void PnPsolver::relative_error(double &rot_err, double &transl_err,
                               const double Rtrue[3][3], const double ttrue[3],
                               const double Rest[3][3], const double test[3]) {
  double qtrue[4], qest[4];

  mat_to_quat(Rtrue, qtrue);
  mat_to_quat(Rest, qest);

  double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
                         (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
                         (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
                         (qtrue[3] - qest[3]) * (qtrue[3] - qest[3])) /
                    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] +
                         qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
                         (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
                         (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
                         (qtrue[3] + qest[3]) * (qtrue[3] + qest[3])) /
                    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] +
                         qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  rot_err = min(rot_err1, rot_err2);

  transl_err =
      sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
           (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
           (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
      sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

/// @brief 旋转矩阵转四元数
/// @param R 旋转矩阵 输入
/// @param q 四元数 输出
void PnPsolver::mat_to_quat(const double R[3][3], double q[4]) {
  double tr = R[0][0] + R[1][1] + R[2][2];
  double n4;

  if (tr > 0.0f) {
    q[0] = R[1][2] - R[2][1];
    q[1] = R[2][0] - R[0][2];
    q[2] = R[0][1] - R[1][0];
    q[3] = tr + 1.0f;
    n4 = q[3];
  } else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
    q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
    q[1] = R[1][0] + R[0][1];
    q[2] = R[2][0] + R[0][2];
    q[3] = R[1][2] - R[2][1];
    n4 = q[0];
  } else if (R[1][1] > R[2][2]) {
    q[0] = R[1][0] + R[0][1];
    q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
    q[2] = R[2][1] + R[1][2];
    q[3] = R[2][0] - R[0][2];
    n4 = q[1];
  } else {
    q[0] = R[2][0] + R[0][2];
    q[1] = R[2][1] + R[1][2];
    q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
    q[3] = R[0][1] - R[1][0];
    n4 = q[2];
  }
  double scale = 0.5f / double(sqrt(n4));

  q[0] *= scale;
  q[1] *= scale;
  q[2] *= scale;
  q[3] *= scale;
}

}  // namespace ORB_SLAM2
