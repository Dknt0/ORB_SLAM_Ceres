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

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/imgproc/imgproc_c.h>

#include <opencv2/opencv.hpp>

#include "Frame.h"
#include "MapPoint.h"

namespace ORB_SLAM2 {

// EPnP 算法
// https://github.com/cvlab-epfl/EPnP/

// Tracking::Relocalization

/// @brief PnP 求解器
class PnPsolver {
 public:
  PnPsolver(const Frame &F, const vector<MapPoint *> &vpMapPointMatches);

  ~PnPsolver();

  void SetRansacParameters(double probability = 0.99, int minInliers = 8,
                           int maxIterations = 300, int minSet = 4,
                           float epsilon = 0.4, float th2 = 5.991);

  cv::Mat find(vector<bool> &vbInliers, int &nInliers);

  cv::Mat iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers,
                  int &nInliers);

 private:
  void CheckInliers();
  bool Refine();

  // Functions from the original EPnP code

  double compute_pose(double R[3][3], double T[3]);

  void set_maximum_number_of_correspondences(const int n);
  void reset_correspondences(void);
  void add_correspondence(const double X, const double Y, const double Z,
                          const double u, const double v);

  void relative_error(double &rot_err, double &transl_err,
                      const double Rtrue[3][3], const double ttrue[3],
                      const double Rest[3][3], const double test[3]);

  void print_pose(const double R[3][3], const double t[3]);
  double reprojection_error(const double R[3][3], const double t[3]);

  void choose_control_points(void);
  void compute_barycentric_coordinates(void);
  void fill_M(CvMat *M, const int row, const double *alphas, const double u,
              const double v);
  void compute_ccs(const double *betas, const double *ut);
  void compute_pcs(void);

  void solve_for_sign(void);

  void find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho,
                           double *betas);
  void find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho,
                           double *betas);
  void find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho,
                           double *betas);
  void qr_solve(CvMat *A, CvMat *b, CvMat *X);

  double dot(const double *v1, const double *v2);
  double dist2(const double *p1, const double *p2);

  void compute_rho(double *rho);
  void compute_L_6x10(const double *ut, double *l_6x10);

  void gauss_newton(const CvMat *L_6x10, const CvMat *Rho,
                    double current_betas[4]);
  void compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
                                    double cb[4], CvMat *A, CvMat *b);

  double compute_R_and_t(const double *ut, const double *betas, double R[3][3],
                         double t[3]);

  void estimate_R_and_t(double R[3][3], double t[3]);

  void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
                    double R_src[3][3], double t_src[3]);

  void mat_to_quat(const double R[3][3], double q[4]);

  // 以下"关联"指 3D-2D 投影关联

  /* EPnP 内部变量 */
  double uc, vc, fu, fv;  // 相机内参
  double *pws;            // 空间点世界坐标  3 * max_n
  double *us;             // 空间点像素坐标  2 * max_n
  double *alphas;         // 空间点对应控制点系数  4 * max_n
  double *pcs;            // 空间点相机坐标  3 * max_n
  int maximum_number_of_correspondences;  // 最大关联数量
  int number_of_correspondences;          // 当前关联数量
  double cws[4][3];                       // 控制点世界坐标
  double ccs[4][3];                       // 控制点相机坐标
  double cws_determinant;                 // 没有用到

  /* 地图点与关键点相关  在 RANSAC, EPnP 计算过程中作为常数使用 */
  // Number of Correspondences
  vector<MapPoint *>
      mvpMapPointMatches;  // 匹配地图点集
                           // 原始输入，包含空指针和坏点指针，需筛选
  // 3D Points
  int N;  // 3D-2D 投影关联总数  有效地图点观测数量
  vector<cv::Point3f>
      mvP3Dw;  // 有效地图点空间位置  无坏点  按照有效地图点顺序索引
  // 2D Points
  vector<cv::Point2f> mvP2D;  // 关键点集  按照有效地图点顺序索引
  vector<float> mvSigma2;  // 尺度比例平方  按照有效地图点顺序索引
  // Index in Frame
  vector<size_t> mvKeyPointIndices;  // 有效地图点 对应 匹配地图点集 id
                                     // 按照有效地图点顺序索引
  // Indices for random selection [0 .. N-1]
  vector<size_t> mvAllIndices;  // RANSAC 采样有效地图点序号
                                // 这个貌似没意义，0中存0，1中存1...

  /* 当前估计值 */
  // Current Estimation
  double mRi[3][3];          // 当前姿态估计
  double mti[3];             // 当前位置估计
  cv::Mat mTcwi;             // 当前位姿
  vector<bool> mvbInliersi;  // 当前内点标志位
  int mnInliersi;            // 当前内点数量

  /* 当前 RANSAC 状态 */
  // Current Ransac State
  int mnIterations;             // RANSAC 迭代总次数
  vector<bool> mvbBestInliers;  // RANSAC 最佳内点数量 对应内点标志位
  int mnBestInliers;            // 当前 RANSAC 最佳内点数量
  cv::Mat mBestTcw;             // RANSAC 最佳内点数量 对应 Tcw

  /* 改善后状态 */
  // Refined
  cv::Mat mRefinedTcw;             // 改善后相机位姿 Tcw
  vector<bool> mvbRefinedInliers;  // 改善后内点
  int mnRefinedInliers;            // 改善后内点数量

  /* RANSAC 相关 */
  // RANSAC probability
  double mRansacProb;  // =0.99 RANSAC 概率
  // RANSAC min inliers
  int mRansacMinInliers;  // =8 RANSAC 最少内点数
  // RANSAC max iterations
  int mRansacMaxIts;  // RANSAC 最大迭代次数
  // RANSAC expected inliers/total ratio
  float mRansacEpsilon;  // =0.4 RANSAC 期望内点/总数比例
  // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
  float mRansacTh;  // RANSAC 内点/总数比例阈值
  // RANSAC Minimun Set used at each iteration
  int mRansacMinSet;  // =4  RANSAC 最小关联数量  求解一次 EPnP 随机添加的关联数
  // Max square error associated with scale level. Max error =
  // th*th*sigma(level)*sigma(level)
  vector<float> mvMaxError;  // 依据尺度级别计算的平方误差阈值 用于内点筛选
};

}  // namespace ORB_SLAM2

#endif  // PNPSOLVER_H
