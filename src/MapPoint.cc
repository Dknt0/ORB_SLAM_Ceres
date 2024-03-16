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

#include "MapPoint.h"

#include <mutex>

#include "ORBmatcher.h"

namespace ORB_SLAM2 {

long unsigned int MapPoint::nNextId = 0;  // 下一帧序号
mutex MapPoint::mGlobalMutex;             // 全局变量锁

/// @brief 地图点构造函数  从关键帧构造地图点
/// @param Pos 位置 tw
/// @param pRefKF 参考关键帧
/// @param pMap 地图
MapPoint::MapPoint(const cv::Mat& Pos, KeyFrame* pRefKF, Map* pMap)
    : mnFirstKFid(pRefKF->mnId),
      mnFirstFrame(pRefKF->mnFrameId),
      nObs(0),
      // 其他线程变量
      mnTrackReferenceForFrame(0),
      mnLastFrameSeen(0),
      mnBALocalForKF(0),
      mnFuseCandidateForKF(0),
      mnLoopPointForKF(0),
      mnCorrectedByKF(0),
      mnCorrectedReference(0),
      mnBAGlobalForKF(0),
      //
      mpRefKF(pRefKF),
      mfMinDistance(0),
      mfMaxDistance(0),
      mnVisible(1),
      mnFound(1),
      mbBad(false),
      mpReplaced(static_cast<MapPoint*>(NULL)),
      mpMap(pMap) {
  /* 设置空间位置 */
  Pos.copyTo(mWorldPos);
  /* 设置观测方向 */
  mNormalVector = cv::Mat::zeros(3, 1, CV_32F);  // 观测方向 零向量

  // 这里没有计算观测方向，没有提供描述子信息

  // MapPoints can be created from Tracking and Local Mapping. This mutex avoid
  // conflicts with id.
  unique_lock<mutex> lock(mpMap->mMutexPointCreation);
  mnId = nNextId++;
}

/// @brief 地图点构造函数  从帧构造地图点
/// 在纯定位模式中基于运动模型计算位姿进行帧间匹配时使用
/// @param Pos 位置
/// @param pMap 地图
/// @param pFrame 帧
/// @param idxF 地图点对应帧内左图关键点序号
MapPoint::MapPoint(const cv::Mat& Pos, Map* pMap, Frame* pFrame,
                   const int& idxF)
    : mnFirstKFid(-1),
      mnFirstFrame(pFrame->mnId),
      nObs(0),
      // 其他线程变量
      mnTrackReferenceForFrame(0),
      mnLastFrameSeen(0),
      mnBALocalForKF(0),
      mnFuseCandidateForKF(0),
      mnLoopPointForKF(0),
      mnCorrectedByKF(0),
      mnCorrectedReference(0),
      mnBAGlobalForKF(0),
      //
      mpRefKF(static_cast<KeyFrame*>(NULL)),
      mnVisible(1),
      mnFound(1),
      mbBad(false),
      mpReplaced(NULL),
      mpMap(pMap) {
  Pos.copyTo(mWorldPos);

  // 计算观测方向
  cv::Mat Ow = pFrame->GetCameraCenter();  // 相机
  mNormalVector = mWorldPos - Ow;          // 相机到地图点
  mNormalVector = mNormalVector / cv::norm(mNormalVector);  // 单位向量

  cv::Mat PC = Pos - Ow;            // 相机到地图点   同 mWorldPos - Ow
  const float dist = cv::norm(PC);  // 距离
  const int level = pFrame->mvKeysUn[idxF].octave;  // 金字塔尺度
  const float levelScaleFactor =
      pFrame->mvScaleFactors[level];          // 金字塔尺度对应比例 >1
  const int nLevels = pFrame->mnScaleLevels;  // 金字塔总层数

  mfMaxDistance =
      dist * levelScaleFactor;  // 最远距离为在 0 层检测到特征点时对应的距离
  mfMinDistance =
      mfMaxDistance /
      pFrame->mvScaleFactors[nLevels -
                             1];  // 最近距离为在最高层检测到特征点时对应的距离

  pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);  // 描述子

  // MapPoints can be created from Tracking and Local Mapping. This mutex avoid
  // conflicts with id.
  unique_lock<mutex> lock(mpMap->mMutexPointCreation);
  mnId = nNextId++;
}

/// @brief 设置世界坐标 tw
/// @param Pos
void MapPoint::SetWorldPos(const cv::Mat& Pos) {
  // 为什么需要全局锁?
  unique_lock<mutex> lock2(mGlobalMutex);
  unique_lock<mutex> lock(mMutexPos);
  Pos.copyTo(mWorldPos);
}

/// @brief 获取世界坐标 tw
/// @return 世界坐标
cv::Mat MapPoint::GetWorldPos() {
  unique_lock<mutex> lock(mMutexPos);
  return mWorldPos.clone();
}

/// @brief 获取平均观测方向
/// @return
cv::Mat MapPoint::GetNormal() {
  unique_lock<mutex> lock(mMutexPos);
  return mNormalVector.clone();
}

/// @brief 获取参考关键帧
/// @return
KeyFrame* MapPoint::GetReferenceKeyFrame() {
  unique_lock<mutex> lock(mMutexFeatures);
  return mpRefKF;
}

/// @brief 添加关键帧观测
/// @param pKF 关键帧
/// @param idx 地图点在关键帧内左图关键点序号
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx) {
  unique_lock<mutex> lock(mMutexFeatures);
  // 如果观测映射中已存在此关键帧，返回
  if (mObservations.count(pKF)) return;
  mObservations[pKF] = idx;

  // 双目点算两次观测
  if (pKF->mvuRight[idx] >= 0)
    nObs += 2;
  else
    nObs++;
}

/// @brief 清除关键帧观测，坏点检测
/// @param pKF 关键帧
void MapPoint::EraseObservation(KeyFrame* pKF) {
  bool bBad = false;
  {
    unique_lock<mutex> lock(mMutexFeatures);
    // 检查观测集中是否有这个关键帧
    if (mObservations.count(pKF)) {
      int idx = mObservations[pKF];  // 地图点对应关键点序号
      // 双目 -2
      if (pKF->mvuRight[idx] >= 0) nObs -= 2;
      // 单目 -1
      else
        nObs--;

      mObservations.erase(pKF);  // 从观测集中删除

      // 如果参考帧是被删除的帧，则将观测集中第一个帧设置为参考帧
      if (mpRefKF == pKF) mpRefKF = mObservations.begin()->first;

      // If only 2 observations or less, discard point
      // 观测数小于等于 2 时设置为坏点
      if (nObs <= 2) bBad = true;
    }
  }

  if (bBad) SetBadFlag();
}

/// @brief 观测关键帧集
/// @return
map<KeyFrame*, size_t> MapPoint::GetObservations() {
  unique_lock<mutex> lock(mMutexFeatures);
  return mObservations;
}

/// @brief 获取 KF 观测次数  如果小于 1 则此 MP 为坏点或纯定位模式中的临时点
/// @return
int MapPoint::Observations() {
  unique_lock<mutex> lock(mMutexFeatures);
  return nObs;
}

/// @brief 设置坏点标志位，从观测关键帧中删除此点，从地图中删除此点
void MapPoint::SetBadFlag() {
  map<KeyFrame*, size_t> obs;
  // 获取观测集
  {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    mbBad = true;
    obs = mObservations;
    mObservations.clear();
  }

  // 遍历观测集 从对应关键帧中删除这个地图点
  for (map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end();
       mit != mend; mit++) {
    KeyFrame* pKF = mit->first;
    pKF->EraseMapPointMatch(
        mit->second);  // 关键帧内 按照关键点索引清除地图点匹配
  }

  // 从地图中删除地图点
  mpMap->EraseMapPoint(this);
}

/// @brief 获取替代点
/// @return
MapPoint* MapPoint::GetReplaced() {
  unique_lock<mutex> lock1(mMutexFeatures);
  unique_lock<mutex> lock2(mMutexPos);
  return mpReplaced;
}

/// @brief 替代地图点  新地图点会继承当前点与关键帧之间的观测关系
/// @param pMP 新地图点
void MapPoint::Replace(MapPoint* pMP) {
  // 如果是同一个地图点，忽略
  if (pMP->mnId == this->mnId) return;

  int nvisible;                // 上一个点可视次数
  int nfound;                  // 上一个点检测次数
  map<KeyFrame*, size_t> obs;  // 观测集
  // 暂存当前地图点，获取信息，然后设置坏点标志
  {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    obs = mObservations;
    mObservations.clear();
    mbBad = true;
    nvisible = mnVisible;
    nfound = mnFound;
    mpReplaced = pMP;  // 当前点被这个点替换
  }

  // 遍历旧点观测集 新点继承旧点观测关系，调整新点观测、调整旧点对应关键帧的观测
  for (map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end();
       mit != mend; mit++) {
    // Replace measurement in keyframe
    KeyFrame* pKF = mit->first;  // 关键帧

    // 观测到旧点的关键帧没有观测到新点，则新点继承这个匹配关系
    if (!pMP->IsInKeyFrame(pKF)) {
      pKF->ReplaceMapPointMatch(mit->second,
                                pMP);  // 将旧地图点对应关键点的匹配设置为新点
      pMP->AddObservation(pKF, mit->second);  // 在新点中添加这个关键帧观测
    }
    // 观测到旧点的关键帧也观测到新点，说明新点对应到了别的关键点，需要清除旧点地图点匹配
    else {
      pKF->EraseMapPointMatch(mit->second);  // 按照关键点索引清除地图点匹配
    }
  }

  pMP->IncreaseFound(nfound);            // 叠加检测次数
  pMP->IncreaseVisible(nvisible);        // 叠加可视次数
  pMP->ComputeDistinctiveDescriptors();  // 计算最佳描述子

  mpMap->EraseMapPoint(this);  // 从地图删除当前点
}

/// @brief 是否为坏点
/// @return
bool MapPoint::isBad() {
  unique_lock<mutex> lock(mMutexFeatures);
  unique_lock<mutex> lock2(mMutexPos);
  return mbBad;
}

/// @brief 叠加可视次数
/// @param n 叠加数
void MapPoint::IncreaseVisible(int n) {
  unique_lock<mutex> lock(mMutexFeatures);
  mnVisible += n;
}

/// @brief 叠加检测次数
/// @param n 叠加数
void MapPoint::IncreaseFound(int n) {
  unique_lock<mutex> lock(mMutexFeatures);
  mnFound += n;
}

/// @brief 获取检测比  检测比=检测次数/可视次数
/// @return 检测比
float MapPoint::GetFoundRatio() {
  unique_lock<mutex> lock(mMutexFeatures);
  return static_cast<float>(mnFound) / mnVisible;
}

/// @brief 计算最佳描述子
void MapPoint::ComputeDistinctiveDescriptors() {
  // Retrieve all observed descriptors
  vector<cv::Mat> vDescriptors;  // 地图点在所有关键帧内对应特征点描述子的集合

  map<KeyFrame*, size_t> observations;  // 观测集

  {
    unique_lock<mutex> lock1(mMutexFeatures);
    if (mbBad) return;
    observations = mObservations;
  }

  if (observations.empty()) return;

  // 描述子数量等于关键帧数量
  vDescriptors.reserve(observations.size());

  // 遍历观测集，从关键帧中获取描述子
  for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                        mend = observations.end();
       mit != mend; mit++) {
    KeyFrame* pKF = mit->first;

    // 描述子已经在关键帧中计算过了，只需要对应行号取过来
    if (!pKF->isBad())
      vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
  }

  if (vDescriptors.empty()) return;

  // Compute distances between them
  const size_t N = vDescriptors.size();

  float Distances[N][N];  // 描述子汉明距离矩阵，是个对称矩阵
  // 遍历每一对描述子，计算距离矩阵
  for (size_t i = 0; i < N; i++) {
    Distances[i][i] = 0;
    for (size_t j = i + 1; j < N; j++) {
      // 计算描述子汉明距离
      int distij =
          ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
      Distances[i][j] = distij;
      Distances[j][i] = distij;
    }
  }

  // Take the descriptor with least median distance to the rest
  // 选用与其他描述子汉明距离中位数最小的描述子作为最佳描述子
  int BestMedian = INT_MAX;
  int BestIdx = 0;
  // 遍历每一行  相当对一个描述子与其他描述子的距离进行排序
  for (size_t i = 0; i < N; i++) {
    vector<int> vDists(Distances[i], Distances[i] + N);  // 取一行
    sort(vDists.begin(), vDists.end());                  // 排序
    int median = vDists[0.5 * (N - 1)];                  // 取中位数

    if (median < BestMedian) {
      BestMedian = median;
      BestIdx = i;
    }
  }

  // 保存最佳描述子
  {
    unique_lock<mutex> lock(mMutexFeatures);
    mDescriptor = vDescriptors[BestIdx].clone();
  }
}

/// @brief 获取最佳描述子
/// @return
cv::Mat MapPoint::GetDescriptor() {
  unique_lock<mutex> lock(mMutexFeatures);
  return mDescriptor.clone();
}

/// @brief 获取此地图点在某关键帧中对应关键点的序号
/// @param pKF 关键帧
/// @return 关键点序号
int MapPoint::GetIndexInKeyFrame(KeyFrame* pKF) {
  unique_lock<mutex> lock(mMutexFeatures);
  if (mObservations.count(pKF))
    return mObservations[pKF];
  else
    return -1;
}

/// @brief 地图点被关键帧观测到
/// @param pKF 关键帧
/// @return
bool MapPoint::IsInKeyFrame(KeyFrame* pKF) {
  unique_lock<mutex> lock(mMutexFeatures);
  return (mObservations.count(pKF));
}

/// @brief 更新地图点平均观测方向和尺度无关距离
void MapPoint::UpdateNormalAndDepth() {
  map<KeyFrame*, size_t> observations;  // 观测集
  KeyFrame* pRefKF;                     // 参考关键帧
  cv::Mat Pos;                          // 位置 tw
  // 获取信息
  {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    if (mbBad) return;
    observations = mObservations;
    pRefKF = mpRefKF;
    Pos = mWorldPos.clone();
  }

  if (observations.empty()) return;

  cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);  // 观测方向
  int n = 0;
  // 遍历所有观测
  for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                        mend = observations.end();
       mit != mend; mit++) {
    KeyFrame* pKF = mit->first;
    cv::Mat Owi = pKF->GetCameraCenter();  // 相机位置
    cv::Mat normali = mWorldPos - Owi;     // 这一帧的观测方向
    normal = normal + normali / cv::norm(normali);
    n++;
  }

  cv::Mat PC = Pos - pRefKF->GetCameraCenter();  // 参考帧观测方向
  const float dist = cv::norm(PC);               // 观测距离
  const int level = pRefKF->mvKeysUn[observations[pRefKF]]
                        .octave;  // 对应关键帧关键点金字塔尺度
  const float levelScaleFactor = pRefKF->mvScaleFactors[level];  // 尺度比例 >1
  const int nLevels = pRefKF->mnScaleLevels;  // 金字塔层数

  {
    unique_lock<mutex> lock3(mMutexPos);
    mfMaxDistance = dist * levelScaleFactor;  // 最远尺度无关距离
    mfMinDistance = mfMaxDistance /
                    pRefKF->mvScaleFactors[nLevels - 1];  // 最近尺度无关距离
    mNormalVector = normal / n;                           // 平均观测方向
  }
}

/// @brief 获取最小尺度无关距离*0.8    1/1.2=0.83333
/// @return
float MapPoint::GetMinDistanceInvariance() {
  unique_lock<mutex> lock(mMutexPos);
  return 0.8f * mfMinDistance;
}

/// @brief 获取最大尺度无关距离*1.2
/// @return
float MapPoint::GetMaxDistanceInvariance() {
  unique_lock<mutex> lock(mMutexPos);
  return 1.2f * mfMaxDistance;
}

/// @brief 根据距离预测金字塔尺度
/// @param currentDist 当前距离
/// @param pKF 关键帧
/// @return 金字塔尺度
int MapPoint::PredictScale(const float& currentDist, KeyFrame* pKF) {
  float ratio;  // 尺度
  {
    unique_lock<mutex> lock(mMutexPos);
    ratio = mfMaxDistance / currentDist;  // 预测比例 >1
  }

  // ratio = scaleFactor^n
  // log(ratio) = log(scaleFactor) * n

  int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
  if (nScale < 0)
    nScale = 0;
  else if (nScale >= pKF->mnScaleLevels)
    nScale = pKF->mnScaleLevels - 1;  // 若预测层数大于最大层数，取最大层数

  return nScale;
}

/// @brief 根据距离预测金字塔尺度
/// @param currentDist 当前距离
/// @param pF 帧
/// @return 金字塔尺度
int MapPoint::PredictScale(const float& currentDist, Frame* pF) {
  float ratio;  // 尺度
  {
    unique_lock<mutex> lock(mMutexPos);
    ratio = mfMaxDistance / currentDist;  // 预测比例 >1
  }

  int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
  if (nScale < 0)
    nScale = 0;
  else if (nScale >= pF->mnScaleLevels)
    nScale = pF->mnScaleLevels - 1;  // 若预测层数大于最大层数，取最大层数

  return nScale;
}

//////////////////////////////////////////////



}  // namespace ORB_SLAM2
