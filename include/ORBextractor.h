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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ORB_SLAM2 {

/// @brief 特征区域划分四叉树节点
class ExtractorNode {
 public:
  ExtractorNode() : bNoMore(false) {}

  void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3,
                  ExtractorNode &n4);

  std::vector<cv::KeyPoint> vKeys;         // 区域内的所有特征点
  cv::Point2i UL, UR, BL, BR;              // 当前分块边界点坐标
  std::list<ExtractorNode>::iterator lit;  // 节点列表迭代器，指向自己
  bool bNoMore;                            // 是否只包含一个特征点
};

// 提取器每次处理之后都会暂存上一次处理的金字塔等数据，可以供搜索等使用

/// @brief 提取器构造函数
class ORBextractor {
 public:
  enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

  ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST,
               int minThFAST);

  ~ORBextractor() {}

  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.

  void operator()(cv::InputArray image, cv::InputArray mask,
                  std::vector<cv::KeyPoint> &keypoints,
                  cv::OutputArray descriptors);

  int inline GetLevels() { return nlevels; }

  float inline GetScaleFactor() { return scaleFactor; }

  std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

  std::vector<float> inline GetInverseScaleFactors() {
    return mvInvScaleFactor;
  }

  std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

  std::vector<float> inline GetInverseScaleSigmaSquares() {
    return mvInvLevelSigma2;
  }

  std::vector<cv::Mat> mvImagePyramid;  // 图像金字塔

 protected:
  void ComputePyramid(cv::Mat image);
  void ComputeKeyPointsOctTree(
      std::vector<std::vector<cv::KeyPoint> > &allKeypoints);
  std::vector<cv::KeyPoint> DistributeOctTree(
      const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
      const int &maxX, const int &minY, const int &maxY, const int &nFeatures,
      const int &level);

  void ComputeKeyPointsOld(
      std::vector<std::vector<cv::KeyPoint> > &allKeypoints);

  // 描述子计算模板
  // cv::Point 是 cv::Point2i 类型
  std::vector<cv::Point> pattern;

  int nfeatures;       // 特征总数
  double scaleFactor;  // 金字塔层间比例
  int nlevels;         // 金字塔层数
  int iniThFAST;       // FAST 初始阈值
  int minThFAST;       // FAST 最小阈值

  std::vector<int> mnFeaturesPerLevel;  // 每一层中的特征点数

  std::vector<int> umax;  // 预先计算的 1/4 圆弧点坐标

  std::vector<float> mvScaleFactor;  // 每一层的绝对尺寸比例  大于 1
  std::vector<float> mvInvScaleFactor;  // 每一层的逆绝对尺寸比例  小于 1
  std::vector<float> mvLevelSigma2;     // 金字塔面积绝对比例  大于 1
  std::vector<float> mvInvLevelSigma2;  // 金字塔逆面积绝对比例  小于 1
};

}  // namespace ORB_SLAM2

#endif
