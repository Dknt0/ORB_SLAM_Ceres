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
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <opencv2/opencv.hpp>

#include "Frame.h"

namespace ORB_SLAM2 {

/// @brief 单目初始化器
class Initializer {
  typedef pair<int, int> Match;  // 匹配对  pair<F1 KP id, F2 KP id>

 public:
  Initializer(const Frame &ReferenceFrame, float sigma = 1.0,
              int iterations = 200);
  bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                  cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                  vector<bool> &vbTriangulated);

 private:
  // RANSAC 采样

  void SampleForRansec();

  // 基础矩阵相关

  void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);
  cv::Mat ComputeF21(const vector<cv::Point2f> &vP1,
                     const vector<cv::Point2f> &vP2);
  float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers,
                         float sigma);
  bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                    vector<bool> &vbTriangulated, float minParallax,
                    int minTriangulated);
  void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

  // 单应矩阵相关

  void FindHomography(std::vector<bool> &vbMatchesInliers, float &score,
                      cv::Mat &H21);
  cv::Mat ComputeH21(const vector<cv::Point2f> &vP1,
                     const vector<cv::Point2f> &vP2);
  float CheckHomography(const cv::Mat &H21, const cv::Mat &H12,
                        vector<bool> &vbMatchesInliers, float sigma);
  bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                    vector<bool> &vbTriangulated, float minParallax,
                    int minTriangulated);

  // 公共函数

  void Normalize(const vector<cv::KeyPoint> &vKeys,
                 vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
  void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                   const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
  int CheckRT(const cv::Mat &R, const cv::Mat &t,
              const vector<cv::KeyPoint> &vKeys1,
              const vector<cv::KeyPoint> &vKeys2,
              const vector<Match> &vMatches12, vector<bool> &vbInliers,
              const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2,
              vector<bool> &vbGood, float &parallax);

  /* 关键点匹配信息 */
  vector<cv::KeyPoint> mvKeys1;  // 参考帧 F1 去畸变 KP
  vector<cv::KeyPoint> mvKeys2;  // 当前帧 F1 去畸变 KP
  vector<Match> mvMatches12;  // 有效匹配对 F1 KP 匹配到 F2 KP  vector<pair<F1
                              // KP id, F2 KP id>>
  vector<bool> mvbMatched1;  // F1 KP 匹配成功标志位
  cv::Mat mK;                // 相机内参
  float mSigma;              // 测量标准差  default 1.0
  float mSigma2;             // 测量方差

  /* RANSAC 相关 */
  int mMaxIterations;              // RANSAC 最大迭代次数  default 200
  vector<vector<size_t> > mvSets;  // RANSAC 样本集  [200][8][有效匹配对序号]
};

}  // namespace ORB_SLAM2

#endif  // INITIALIZER_H
