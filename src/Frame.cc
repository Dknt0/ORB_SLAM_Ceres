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

#include "Frame.h"

#include <thread>

#include "Converter.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2 {

// 静态变量
long unsigned int Frame::nNextId = 0;      // 下一帧序号
bool Frame::mbInitialComputations = true;  // 是否在初始化

// 相机内参
float Frame::cx;     // 相机内参  x 偏移
float Frame::cy;     // 相机内参  y 偏移
float Frame::fx;     // 相机内参  x 焦距
float Frame::fy;     // 相机内参  y 焦距
float Frame::invfx;  // 相机内参  x 逆焦距
float Frame::invfy;  // 相机内参  x 逆焦距

// 去畸变图像实际范围，最大程度保留图像信息，按照畸变类型不同，边界可能位于图像外
float Frame::mnMinX;  // 去畸变后左边界
float Frame::mnMinY;  // 去畸变后上边界
float Frame::mnMaxX;  // 去畸变后右边界
float Frame::mnMaxY;  // 去畸变后下边界

float Frame::mfGridElementWidthInv;   // 网格宽度倒数
float Frame::mfGridElementHeightInv;  // 网格高度倒数

/// @brief 默认构造函数
Frame::Frame() {}

/// @brief 拷贝构造函数  注意 id 也是拷贝的
/// @param frame
Frame::Frame(const Frame &frame)
    : mpORBvocabulary(frame.mpORBvocabulary),
      mpORBextractorLeft(frame.mpORBextractorLeft),
      mpORBextractorRight(frame.mpORBextractorRight),
      mnId(frame.mnId),
      mTimeStamp(frame.mTimeStamp),
      mpReferenceKF(frame.mpReferenceKF),
      mK(frame.mK.clone()),
      mDistCoef(frame.mDistCoef.clone()),
      mbf(frame.mbf),
      mb(frame.mb),
      mThDepth(frame.mThDepth),
      N(frame.N),
      mvKeys(frame.mvKeys),
      mvKeysRight(frame.mvKeysRight),
      mvKeysUn(frame.mvKeysUn),
      mvuRight(frame.mvuRight),
      mvDepth(frame.mvDepth),
      mDescriptors(frame.mDescriptors.clone()),
      mDescriptorsRight(frame.mDescriptorsRight.clone()),
      mvpMapPoints(frame.mvpMapPoints),
      mvbOutlier(frame.mvbOutlier),
      mBowVec(frame.mBowVec),
      mFeatVec(frame.mFeatVec),
      mnScaleLevels(frame.mnScaleLevels),
      mfScaleFactor(frame.mfScaleFactor),
      mfLogScaleFactor(frame.mfLogScaleFactor),
      mvScaleFactors(frame.mvScaleFactors),
      mvInvScaleFactors(frame.mvInvScaleFactors),
      mvLevelSigma2(frame.mvLevelSigma2),
      mvInvLevelSigma2(frame.mvInvLevelSigma2) {
  // 拷贝网格信息
  for (int i = 0; i < FRAME_GRID_COLS; i++)
    for (int j = 0; j < FRAME_GRID_ROWS; j++) mGrid[i][j] = frame.mGrid[i][j];

  // 如果有位姿信息，拷贝
  if (!frame.mTcw.empty()) SetPose(frame.mTcw);
}

/// @brief 双目构造函数
/// @param imLeft 左图
/// @param imRight 右图
/// @param timeStamp 时间戳
/// @param extractorLeft 左提取器
/// @param extractorRight 右提取器
/// @param voc 视觉词典
/// @param K 内参矩阵
/// @param distCoef 畸变参数
/// @param bf 基线 * 焦距
/// @param thDepth 深度阈值
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight,
             const double &timeStamp, ORBextractor *extractorLeft,
             ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K,
             cv::Mat &distCoef, const float &bf, const float &thDepth)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(extractorLeft),
      mpORBextractorRight(extractorRight),
      mTimeStamp(timeStamp),
      mpReferenceKF(static_cast<KeyFrame *>(NULL)),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),  // cv::Mat 注意要深拷贝
      mbf(bf),
      mThDepth(thDepth) {
  // Frame ID
  mnId = nNextId++;

  /* 1.获取金字塔尺度信息 */

  // Scale Level Info
  // 从左提取器获取金字塔尺度信息
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  /* 2.特征提取与描述子计算 */

  // ORB extraction
  // 双线程提特征
  thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
  thread threadRight(&Frame::ExtractORB, this, 1, imRight);
  threadLeft.join();
  threadRight.join();

  N = mvKeys.size();

  if (mvKeys.empty()) return;

  /* 3.畸变校正 */
  UndistortKeyPoints();

  /* 4.匹配双目特征点，计算右点坐标 */
  ComputeStereoMatches();

  mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
  mvbOutlier = vector<bool>(N, false);

  /* 5.计算去畸变图像边界 */

  // This is done only for the first Frame (or after a change in the
  // calibration) 初始化时调用
  if (mbInitialComputations) {
    // 计算边界
    ComputeImageBounds(imLeft);

    // 网格宽度倒数
    mfGridElementWidthInv =
        static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
    // 网格高度倒数
    mfGridElementHeightInv =
        static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  /* 6.分配特征到网格 */
  AssignFeaturesToGrid();
}

/// @brief 深度图构造函数
/// @param imGray 灰度图
/// @param imDepth 深度图
/// @param timeStamp 时间戳
/// @param extractor 提取器
/// @param voc 视觉字典
/// @param K 内参矩阵
/// @param distCoef 畸变参数
/// @param bf 基线 * 焦距
/// @param thDepth 深度阈值
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth,
             const double &timeStamp, ORBextractor *extractor,
             ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
             const float &thDepth)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(extractor),
      mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
      mTimeStamp(timeStamp),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mbf(bf),
      mThDepth(thDepth) {
  // Frame ID
  mnId = nNextId++;

  /* 1.获取金字塔尺度信息 */

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  /* 2.特征提取与描述子计算 */

  // ORB extraction
  // 特征提取
  ExtractORB(0, imGray);

  N = mvKeys.size();

  if (mvKeys.empty()) return;

  /* 3.畸变校正 */

  // 去畸变
  UndistortKeyPoints();

  /* 4.从深度图像计算视差 */

  // 从深度计算视差
  ComputeStereoFromRGBD(imDepth);

  mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
  mvbOutlier = vector<bool>(N, false);

  /* 5.计算去畸变图像边界 */

  // This is done only for the first Frame (or after a change in the
  // calibration) 初始化时调用
  if (mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                            static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                             static_cast<float>(mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  /* 6.分配特征到网格 */

  // 分配特征到网格
  AssignFeaturesToGrid();
}

/// @brief 单目图像构造函数
/// @param imGray 灰度图
/// @param timeStamp 时间戳
/// @param extractor 提取器
/// @param voc 视觉词典
/// @param K 内参矩阵
/// @param distCoef 畸变参数
/// @param bf 基线 * 焦距
/// @param thDepth 深度阈值  单目用得到这个???
Frame::Frame(const cv::Mat &imGray, const double &timeStamp,
             ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K,
             cv::Mat &distCoef, const float &bf, const float &thDepth)
    : mpORBvocabulary(voc),
      mpORBextractorLeft(extractor),
      mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
      mTimeStamp(timeStamp),
      mK(K.clone()),
      mDistCoef(distCoef.clone()),
      mbf(bf),
      mThDepth(thDepth) {
  // Frame ID
  mnId = nNextId++;

  /* 1.获取金字塔尺度信息 */

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  /* 2.特征提取与描述子计算 */

  // ORB extraction
  // 特征提取
  ExtractORB(0, imGray);

  N = mvKeys.size();

  if (mvKeys.empty()) return;

  /* 3.畸变校正 */
  UndistortKeyPoints();

  /* 4.右点坐标与深度赋 -1 */

  // Set no stereo information
  // 无双目信息
  mvuRight = vector<float>(N, -1);
  mvDepth = vector<float>(N, -1);

  mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
  mvbOutlier = vector<bool>(N, false);

  /* 5.计算去畸变图像边界 */
  // This is done only for the first Frame (or after a change in the
  // calibration) 初始化时调用
  if (mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) /
                            static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) /
                             static_cast<float>(mnMaxY - mnMinY);

    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
    invfx = 1.0f / fx;
    invfy = 1.0f / fy;

    mbInitialComputations = false;
  }

  mb = mbf / fx;

  /* 6.分配特征到网格 */
  AssignFeaturesToGrid();
}

/// @brief 分配特征到网格
void Frame::AssignFeaturesToGrid() {
  // 预分配向量空间  这个预分配大小是不是不合理？
  int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
      mGrid[i][j].reserve(nReserve);

  // 遍历左图去畸变特征点，分配到网格
  for (int i = 0; i < N; i++) {
    const cv::KeyPoint &kp = mvKeysUn[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

/// @brief 调用提取器，提取特征，计算描述子
/// @param flag 标志位  右图 0, 左图 1
/// @param im 灰度图像
void Frame::ExtractORB(int flag, const cv::Mat &im) {
  if (flag == 0)
    (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
  else
    (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
}

/// @brief 设置位姿
/// @param Tcw
void Frame::SetPose(cv::Mat Tcw) {
  mTcw = Tcw.clone();
  UpdatePoseMatrices();
}

/// @brief 从位姿计算旋转、平移、光心位置等
void Frame::UpdatePoseMatrices() {
  mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
  mRwc = mRcw.t();
  mtcw = mTcw.rowRange(0, 3).col(3);
  mOw = -mRcw.t() * mtcw;
}

/// @brief 检查地图点是否在相机截锥体内，并填充地图点投影到像素平面上的坐标
/// @param pMP 地图点
/// @param viewingCosLimit 夹角余弦阈值
/// @return
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
  pMP->mbTrackInView = false;

  /* 1.计算地图点相对于相机位置 tc */

  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos();  // 地图点绝对坐标

  // 3D in camera coordinates
  const cv::Mat Pc = mRcw * P + mtcw;  // 地图点相对于相机坐标
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  /* 2.深度检查 */

  // Check positive depth
  // 若深度为负，不再范围内
  if (PcZ < 0.0f) return false;

  /* 3.投影点外点检查 */

  // Project in image and check it is not outside
  // 投影到成像平面，检查是否为外点
  const float invz = 1.0f / PcZ;  // 逆深度
  const float u = fx * PcX * invz + cx;
  const float v = fy * PcY * invz + cy;

  if (u < mnMinX || u > mnMaxX) return false;
  if (v < mnMinY || v > mnMaxY) return false;

  /* 4.尺度无关范围检查 */

  // Check distance is in the scale invariance region of the MapPoint
  // 检查距离是否在尺度无关范围内
  const float maxDistance =
      pMP->GetMaxDistanceInvariance();  // 最大尺度不变距离*1.2
  const float minDistance =
      pMP->GetMinDistanceInvariance();  // 最小尺度不变距离*0.8
  const cv::Mat PO = P - mOw;           // 光心到空间点
  const float dist = cv::norm(PO);      // 距离

  if (dist < minDistance || dist > maxDistance) return false;

  /* 5.夹角余弦检查 */

  // Check viewing angle
  // 检查夹角余弦
  cv::Mat Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit) return false;

  /* 6.尺度预测，填充信息 */

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  // 填充地图点信息
  pMP->mbTrackInView = true;
  pMP->mTrackProjX = u;
  pMP->mTrackProjXR = u - mbf * invz;
  pMP->mTrackProjY = v;
  pMP->mnTrackScaleLevel = nPredictedLevel;
  pMP->mTrackViewCos = viewCos;

  return true;
}

/// @brief 获取方形区域中的特征，没错是方形
/// @param x 中心 u
/// @param y 中心 v
/// @param r 一半宽
/// @param minLevel 最小金字塔级别
/// @param maxLevel 最大金字塔级别
/// @return 关键点序号集
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y,
                                        const float &r, const int minLevel,
                                        const int maxLevel) const {
  vector<size_t> vIndices;  // 范围特征点序号
  vIndices.reserve(N);

  // 计算搜索区域包含到的网格范围
  const int nMinCellX =
      max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
  // 如果搜索范围大于边界范围，返回，下同
  if (nMinCellX >= FRAME_GRID_COLS) return vIndices;

  const int nMaxCellX =
      min((int)FRAME_GRID_COLS - 1,
          (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0) return vIndices;

  const int nMinCellY =
      max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS) return vIndices;

  const int nMaxCellY =
      min((int)FRAME_GRID_ROWS - 1,
          (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0) return vIndices;

  // 标志位，是否需要筛选层数  如果用户给的层数为负，则不筛选层数
  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  // 遍历范围内网格行
  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    // 遍历范围内网格列
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const vector<size_t> vCell = mGrid[ix][iy];
      if (vCell.empty()) continue;

      // 遍历网格内特征点
      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];  // 特征点
        // 筛选层数
        if (bCheckLevels) {
          if (kpUn.octave < minLevel) continue;
          if (maxLevel >= 0)
            if (kpUn.octave > maxLevel) continue;
        }

        const float distx = kpUn.pt.x - x;  // x 距离
        const float disty = kpUn.pt.y - y;  // y 距离

        // 这是个方形范围...
        if (fabs(distx) < r && fabs(disty) < r) vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

/// @brief 特征点所在网格位置
/// @param kp 关键点
/// @param posX 网格x 返回值
/// @param posY 网格y 返回值
/// @return 是否在网格内
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
  posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
  posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

  // Keypoint's coordinates are undistorted, which could cause to go out of the
  // image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 ||
      posY >= FRAME_GRID_ROWS)
    return false;

  return true;
}

/// @brief 计算词袋表达
void Frame::ComputeBoW() {
  if (mBowVec.empty()) {
    // 描述子矩阵转向量
    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    // 计算计算词袋表达
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

/// @brief 计算去畸变左图特征点，结果存放于 mvKeysUn
void Frame::UndistortKeyPoints() {
  // 无畸变情况
  if (mDistCoef.at<float>(0) == 0.0) {
    mvKeysUn = mvKeys;
    return;
  }

  // Fill matrix with points
  // 将左图畸变关键点存为 cv::Mat
  cv::Mat mat(N, 2, CV_32F);
  for (int i = 0; i < N; i++) {
    mat.at<float>(i, 0) = mvKeys[i].pt.x;
    mat.at<float>(i, 1) = mvKeys[i].pt.y;
  }

  // Undistort points
  mat = mat.reshape(2);  // 改变通道数为 2
  cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
  mat = mat.reshape(1);  // 改变通道数为 1

  // Fill undistorted keypoint vector
  mvKeysUn.resize(N);
  for (int i = 0; i < N; i++) {
    cv::KeyPoint kp = mvKeys[i];
    kp.pt.x = mat.at<float>(i, 0);
    kp.pt.y = mat.at<float>(i, 1);
    mvKeysUn[i] = kp;
  }
}

/// @brief 计算去畸变后图像的边界点
/// @param imLeft
void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
  // 如果存在畸变
  if (mDistCoef.at<float>(0) != 0.0) {
    // 将畸变图像边界点存为 cv::Mat
    cv::Mat mat(4, 2, CV_32F);
    mat.at<float>(0, 0) = 0.0;
    mat.at<float>(0, 1) = 0.0;  // 左上
    mat.at<float>(1, 0) = imLeft.cols;
    mat.at<float>(1, 1) = 0.0;  // 右上
    mat.at<float>(2, 0) = 0.0;
    mat.at<float>(2, 1) = imLeft.rows;  // 左下
    mat.at<float>(3, 0) = imLeft.cols;
    mat.at<float>(3, 1) = imLeft.rows;  // 右下

    // Undistort corners
    // 对边界进行畸变矫正，即计算畸变点在无畸变图像上的坐标
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));  // 左
    mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));  // 右
    mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));  // 上
    mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));  // 下

  } else {
    // 如果没有畸变，则与原图像重合

    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
  }
}

/// @brief 匹配双目特征点，计算视差
void Frame::ComputeStereoMatches() {
  // 预分配内存
  mvuRight = vector<float>(N, -1.0f);  // 右图 x 坐标
  mvDepth = vector<float>(N, -1.0f);   // 深度

  const int thOrbDist =
      (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;  // ORB 匹配汉明距离阈值

  const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;  // 图像行数

  /* 1.筛选右图候选特征点 */

  // Assign keypoints to row table
  // 按行分割右图关键点
  // 左图第 i 行像素在右图中的候选关键点，跨越了右图第 i 行以及相邻行
  vector<vector<size_t> > vRowIndices(
      nRows, vector<size_t>());  // 候选右图关键点
                                 // 左图行号<候选特征点序号<右图特征点序号>>

  // 预分配内存
  for (int i = 0; i < nRows; i++) vRowIndices[i].reserve(200);

  const int Nr = mvKeysRight.size();  // 右图关键点数

  // 遍历右图关键点，统计候选关键点
  for (int iR = 0; iR < Nr; iR++) {
    const cv::KeyPoint &kp = mvKeysRight[iR];  // 右图关键点
    const float &kpY = kp.pt.y;                // 右关键点 y
    // 考虑到特征点在 v 方向上可能存在偏移，将同一个特征点关联到不同行中
    // 第 0 层关键点有两个像素的纵向偏移，层数越高，跨越越多
    const float r =
        2.0f * mvScaleFactors[mvKeysRight[iR].octave];  // 根据尺度确定跨度
    const int maxr = ceil(kpY + r);
    const int minr = floor(kpY - r);

    for (int yi = minr; yi <= maxr; yi++)
      vRowIndices[yi].push_back(iR);  // 跨越不同行
  }

  // Set limits for search
  // 视差限制与层数无关，只与内参有关
  const float minZ = mb;          // 最小深度
  const float minD = 0;           // 最小视差为 0
  const float maxD = mbf / minZ;  // 最大视差为基线长度

  // For each left keypoint search a match in the right image
  vector<pair<int, int> > vDistIdx;  // <块匹配距离, 左点序号>
  vDistIdx.reserve(N);

  /* 2.特征点匹配 */

  // 遍历左图特征点 寻找匹配
  for (int iL = 0; iL < N; iL++) {
    const cv::KeyPoint &kpL = mvKeys[iL];  // 左图关键点
    const int &levelL = kpL.octave;        // 左特征点金字塔层数
    const float &vL = kpL.pt.y;            // 左特征点 v
    const float &uL = kpL.pt.x;            // 左特征点 u

    const vector<size_t> &vCandidates = vRowIndices[vL];  // 左图 v 行候选特征点

    if (vCandidates.empty()) continue;

    // 右图中特征点应该位于左图 u 左侧，且视差应在一定范围内
    const float minU = uL - maxD;  // 右特征点最小 u
    const float maxU = uL - minD;  // 右特征点最大 u

    // 如果太左，不存在匹配
    if (maxU < 0) continue;

    int bestDist = ORBmatcher::TH_HIGH;  // 最小汉明距离
    size_t bestIdxR = 0;

    const cv::Mat &dL = mDescriptors.row(iL);  // 左特征点描述子

    // Compare descriptor to right keypoints
    // 遍历对应行右图候选特征点，确定匹配
    for (size_t iC = 0; iC < vCandidates.size(); iC++) {
      const size_t iR = vCandidates[iC];          // 右图特征点序号
      const cv::KeyPoint &kpR = mvKeysRight[iR];  // 右图特征点

      // 金字塔层数相差不能超过 1
      if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1) continue;

      const float &uR = kpR.pt.x;  // 右特征点 u

      // 判断视差是否在范围内
      if (uR >= minU && uR <= maxU) {
        const cv::Mat &dR = mDescriptorsRight.row(iR);  // 右特征点描述子
        const int dist =
            ORBmatcher::DescriptorDistance(dL, dR);  // 计算汉明距离

        // 取汉明距离最小的匹配点对
        if (dist < bestDist) {
          bestDist = dist;
          bestIdxR = iR;
        }
      }
    }

    /* 3.滑动窗口像素块匹配 SAD 法 */

    // Subpixel match by correlation
    // 如果最佳匹配汉明距离小于阈值，则进行亚像素修正
    // 亚像素修正仅对最佳匹配对进行
    if (bestDist < thOrbDist) {
      // coordinates in image pyramid at keypoint scale
      // 将左右点坐标都变换到左点对应层上
      const float uR0 = mvKeysRight[bestIdxR].pt.x;  // 右点 0层 u坐标
      const float scaleFactor = mvInvScaleFactors[kpL.octave];  // 左点逆尺度
      const float scaleduL =
          round(kpL.pt.x * scaleFactor);  // 左点金字塔对应层上 u坐标
      const float scaledvL =
          round(kpL.pt.y * scaleFactor);  // 左点金字塔对应层上 v坐标
      const float scaleduR0 =
          round(uR0 * scaleFactor);  // 右点在左点金字塔对应层上 u坐标

      // sliding window search
      // 滑动窗口搜索  只横向滑
      const int w = 5;  // 窗口大小为 11*11
      // 从左图左点对应金字塔层中提取图像块
      cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave]
                       .rowRange(scaledvL - w, scaledvL + w + 1)
                       .colRange(scaleduL - w, scaleduL + w + 1);
      IL.convertTo(IL, CV_32F);  // uchar 转 float
      IL = IL - IL.at<float>(w, w) *
                    cv::Mat::ones(IL.rows, IL.cols, CV_32F);  // 去中心值

      int bestDist = INT_MAX;  // 最佳匹配距离
      int bestincR = 0;        // 最佳块匹配对应偏差
      const int L = 5;         // 滑动窗口范围 -5 ~ +5
      vector<float>
          vDists;  // 所有匹配距离，用于亚像素进度匹配  共 11 个匹配对象
      vDists.resize(2 * L + 1);

      const float iniu = scaleduR0 + L - w;      // 左端 u
      const float endu = scaleduR0 + L + w + 1;  // 右端 u
      // 超出范围则忽略
      if (iniu < 0 ||
          endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
        continue;

      // 遍历
      for (int incR = -L; incR <= +L; incR++) {
        // 取右图中图像块   注意!!这个块的行是按照左点的行取的
        cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave]
                         .rowRange(scaledvL - w, scaledvL + w + 1)
                         .colRange(scaleduR0 + incR - w,
                                   scaleduR0 + incR + w + 1);  // 右图图像块
        // 格式转换，去中心值
        IR.convertTo(IR, CV_32F);
        IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

        // SAD  Sum of absolute differences
        float dist = cv::norm(IL, IR, cv::NORM_L1);  // 图像块 L1 范数
        // 寻找最小匹配距离
        if (dist < bestDist) {
          bestDist = dist;
          bestincR = incR;
        }

        // 记录距离
        vDists[L + incR] = dist;
      }

      // 如果最佳匹配块位于边界，忽略  这里的忽略会导致匹配对弃用吗？
      if (bestincR == -L || bestincR == L) continue;

      /* 4.抛物线拟合法亚像素精度匹配 */

      // Sub-pixel match (Parabola fitting)
      // 亚像素精度匹配  抛物线拟合
      const float dist1 = vDists[L + bestincR - 1];
      const float dist2 = vDists[L + bestincR];
      const float dist3 = vDists[L + bestincR + 1];

      // 抛物线极小值与 dist2 的偏差   公式源自论文   <<On Building an Accurate
      // Stereo Matching System on Graphics Hardware>> 公式7
      const float deltaR =
          (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

      // 如果极小值位于三点之外
      if (deltaR < -1 || deltaR > 1) continue;

      // Re-scaled coordinate
      // 计算在第 0 层上的 u 坐标
      float bestuR = mvScaleFactors[kpL.octave] *
                     ((float)scaleduR0 + (float)bestincR + deltaR);

      float disparity = (uL - bestuR);  // 视差

      if (disparity >= minD && disparity < maxD) {
        // 怎么会小于 0 呢...
        if (disparity <= 0) {
          disparity = 0.01;
          bestuR = uL - 0.01;
        }
        mvDepth[iL] = mbf / disparity;
        mvuRight[iL] = bestuR;
        vDistIdx.push_back(pair<int, int>(bestDist, iL));
      }
    }
  }

  /* 5.匹配点对筛选 */

  // 将匹配结果按照块匹配距离排序
  sort(vDistIdx.begin(), vDistIdx.end());
  const float median = vDistIdx[vDistIdx.size() / 2].first;  // 块匹配距离中值
  const float thDist = 1.5f * 1.4f * median;  // 块匹配距离阈值

  // 匹配筛选
  for (int i = vDistIdx.size() - 1; i >= 0; i--) {
    if (vDistIdx[i].first < thDist)
      break;
    else {
      mvuRight[vDistIdx[i].second] = -1;
      mvDepth[vDistIdx[i].second] = -1;
    }
  }
}

/// @brief 对于深度相机特征点，从深度计算右点坐标
/// @param imDepth 深度图像  float 米
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
  mvuRight = vector<float>(N, -1);
  mvDepth = vector<float>(N, -1);

  // 遍历左图特征点
  for (int i = 0; i < N; i++) {
    const cv::KeyPoint &kp = mvKeys[i];
    const cv::KeyPoint &kpU = mvKeysUn[i];

    // 左点坐标
    const float &v = kp.pt.y;
    const float &u = kp.pt.x;

    const float d = imDepth.at<float>(v, u);

    // 如果深度有效
    if (d > 0) {
      mvDepth[i] = d;
      // 计算假设的右点坐标
      mvuRight[i] = kpU.pt.x - mbf / d;  // 这里用了去畸变坐标
    }
  }
}

/// @brief 由双目信息计算特征点空间位置 tw
/// @param i 特征点索引
/// @return 三维位置向量
cv::Mat Frame::UnprojectStereo(const int &i) {
  const float z = mvDepth[i];
  if (z > 0) {
    const float u = mvKeysUn[i].pt.x;
    const float v = mvKeysUn[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
    return mRwc * x3Dc + mOw;
  } else
    return cv::Mat();
}

/////////////////////////////////////////////////////////////////
void Frame::UpdateEigenPose() {
  Eigen::Isometry3d Tcw_eigen;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      Tcw_eigen(i, j) = mTcw.at<float>(i, j);
    }
  }
  tcw_ = Tcw_eigen.translation();
  qcw_ = Eigen::Quaterniond(Tcw_eigen.rotation());
}

void Frame::UpdatePoseFromEigen() {
  Eigen::Isometry3d Tcw_eigen(qcw_);
  Tcw_eigen.pretranslate(tcw_);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      mTcw.at<float>(i, j) = Tcw_eigen(i, j);
    }
  }
}

}  // namespace ORB_SLAM2
