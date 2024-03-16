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

#include "KeyFrame.h"

#include <mutex>

#include "Converter.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2 {

long unsigned int KeyFrame::nNextId = 0;  // 下一关键帧序号

/// @brief 关键帧构造函数
/// @param F 帧
/// @param pMap 地图
/// @param pKFDB 关键帧数据库
KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB)
    : mnFrameId(F.mnId),
      mTimeStamp(F.mTimeStamp),
      mnGridCols(FRAME_GRID_COLS),
      mnGridRows(FRAME_GRID_ROWS),
      mfGridElementWidthInv(F.mfGridElementWidthInv),
      mfGridElementHeightInv(F.mfGridElementHeightInv),
      mnTrackReferenceForFrame(0),
      mnFuseTargetForKF(0),
      mnBALocalForKF(0),
      mnBAFixedForKF(0),

      mnRelocQuery(0),
      mnLoopWords(0),
      mnLoopQuery(0),
      mnRelocWords(0),
      mnBAGlobalForKF(0),

      fx(F.fx),
      fy(F.fy),
      cx(F.cx),
      cy(F.cy),
      invfx(F.invfx),
      invfy(F.invfy),
      mbf(F.mbf),
      mb(F.mb),
      mThDepth(F.mThDepth),

      N(F.N),
      mvKeys(F.mvKeys),
      mvKeysUn(F.mvKeysUn),
      mvuRight(F.mvuRight),
      mvDepth(F.mvDepth),
      mDescriptors(F.mDescriptors.clone()),
      mBowVec(F.mBowVec),
      mFeatVec(F.mFeatVec),
      mnScaleLevels(F.mnScaleLevels),
      mfScaleFactor(F.mfScaleFactor),
      mfLogScaleFactor(F.mfLogScaleFactor),
      mvScaleFactors(F.mvScaleFactors),
      mvLevelSigma2(F.mvLevelSigma2),
      mvInvLevelSigma2(F.mvInvLevelSigma2),
      mnMinX(F.mnMinX),
      mnMinY(F.mnMinY),
      mnMaxX(F.mnMaxX),
      mnMaxY(F.mnMaxY),
      mK(F.mK),

      mvpMapPoints(F.mvpMapPoints),
      mpKeyFrameDB(pKFDB),
      mpORBvocabulary(F.mpORBvocabulary),

      mpParent(NULL),
      mbFirstConnection(true),
      mbNotErase(false),
      mbToBeErased(false),
      mbBad(false),
      mHalfBaseline(F.mb / 2),
      mpMap(pMap) {
  mnId = nNextId++;

  // 从 Frame 复制特征点网格
  mGrid.resize(mnGridCols);
  for (int i = 0; i < mnGridCols; i++) {
    mGrid[i].resize(mnGridRows);
    for (int j = 0; j < mnGridRows; j++) mGrid[i][j] = F.mGrid[i][j];
  }

  // 设置位姿信息
  SetPose(F.mTcw);
}

/// @brief 计算词袋表达
void KeyFrame::ComputeBoW() {
  if (mBowVec.empty() || mFeatVec.empty()) {
    // 描述子矩阵转向量
    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    // Feature vector associate features with nodes in the 4th level (from
    // leaves up) We assume the vocabulary tree has 6 levels, change the 4
    // otherwise

    // 计算词袋表达
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

/////////////////////////////////////////////////////////////////// 位姿相关函数

/// @brief 设置位姿信息  包括旋转、位置、光心位置等信息
/// @param Tcw_
void KeyFrame::SetPose(const cv::Mat &Tcw_) {
  unique_lock<mutex> lock(mMutexPose);
  Tcw_.copyTo(Tcw);  // 矩阵注意深拷贝
  cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
  cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
  cv::Mat Rwc = Rcw.t();
  Ow = -Rwc * tcw;

  Twc = cv::Mat::eye(4, 4, Tcw.type());
  Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
  Ow.copyTo(Twc.rowRange(0, 3).col(3));
  cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
  Cw = Twc * center;
}

/// @brief 获取位姿 Tcw
/// @return
cv::Mat KeyFrame::GetPose() {
  unique_lock<mutex> lock(mMutexPose);
  return Tcw.clone();
}

/// @brief 获取位姿 Twc
/// @return
cv::Mat KeyFrame::GetPoseInverse() {
  unique_lock<mutex> lock(mMutexPose);
  return Twc.clone();
}

/// @brief 获取位置 twc
/// @return
cv::Mat KeyFrame::GetCameraCenter() {
  unique_lock<mutex> lock(mMutexPose);
  return Ow.clone();
}

/// @brief 获取双目中点 twco
/// @return
cv::Mat KeyFrame::GetStereoCenter() {
  unique_lock<mutex> lock(mMutexPose);
  return Cw.clone();
}

/// @brief 获取姿态 Rcw
/// @return
cv::Mat KeyFrame::GetRotation() {
  unique_lock<mutex> lock(mMutexPose);
  return Tcw.rowRange(0, 3).colRange(0, 3).clone();
}

/// @brief 获取位置 tcw
/// @return
cv::Mat KeyFrame::GetTranslation() {
  unique_lock<mutex> lock(mMutexPose);
  return Tcw.rowRange(0, 3).col(3).clone();
}

///////////////////////////////////////////////////////////////////
///共视图相关函数

/// @brief 添加关键帧共视，更新共视数
/// @param pKF 共视关键帧
/// @param weight 权重
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight) {
  {
    unique_lock<mutex> lock(mMutexConnections);
    // 如果添加关系不存在，创建
    if (!mConnectedKeyFrameWeights.count(pKF))
      mConnectedKeyFrameWeights[pKF] = weight;
    // 如果添加存在但权重不一致，更新权重
    else if (mConnectedKeyFrameWeights[pKF] != weight)
      mConnectedKeyFrameWeights[pKF] = weight;
    else
      return;
  }

  // 更新最佳共视关系
  UpdateBestCovisibles();
}

/// @brief 删除关键帧共视
/// @param pKF 被删除的关键帧
void KeyFrame::EraseConnection(KeyFrame *pKF) {
  bool bUpdate = false;  // 更新标志位
  {
    unique_lock<mutex> lock(mMutexConnections);
    // 如果共视关系存在，清除
    if (mConnectedKeyFrameWeights.count(pKF)) {
      mConnectedKeyFrameWeights.erase(pKF);
      bUpdate = true;  // 更新
    }
  }
  // 更新最佳共视关系
  if (bUpdate) UpdateBestCovisibles();
}

/// @brief 根据已有的共视关键帧，更新按排序的共视关系
void KeyFrame::UpdateBestCovisibles() {
  unique_lock<mutex> lock(mMutexConnections);
  vector<pair<int, KeyFrame *> >
      vPairs;  // 共视数与关键帧集   pair<int,KeyFrame*> 方便排序
  vPairs.reserve(mConnectedKeyFrameWeights.size());  // 预分配内存
  // 遍历共视映射，填充向量
  for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                                      mend = mConnectedKeyFrameWeights.end();
       mit != mend; mit++)
    vPairs.push_back(make_pair(mit->second, mit->first));  // 填充向量

  sort(vPairs.begin(), vPairs.end());  // 按共视数排序
  list<KeyFrame *> lKFs;               // 关键帧列表
  list<int> lWs;                       // 共视数列表
  for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  // 列表转向量
  mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
  mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

/// @brief 获取共视关键帧集合
/// @return 全部共视 KF
set<KeyFrame *> KeyFrame::GetConnectedKeyFrames() {
  unique_lock<mutex> lock(mMutexConnections);
  set<KeyFrame *> s;  // 共视关键帧集合
  for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin();
       mit != mConnectedKeyFrameWeights.end(); mit++)
    s.insert(mit->first);
  return s;
}

/// @brief 获取有序共视关键帧向量  仅包含共视大于 15 的 KF
/// @return
vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames() {
  unique_lock<mutex> lock(mMutexConnections);
  return mvpOrderedConnectedKeyFrames;
}

/// @brief 获取 N 个最佳共视关键帧
/// @param N
/// @return
vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N) {
  unique_lock<mutex> lock(mMutexConnections);
  if ((int)mvpOrderedConnectedKeyFrames.size() < N)
    return mvpOrderedConnectedKeyFrames;
  else
    return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(),
                              mvpOrderedConnectedKeyFrames.begin() + N);
}

/// @brief 获取共视多于 w 的关键帧
/// @param w 共视数
/// @return
vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w) {
  unique_lock<mutex> lock(mMutexConnections);

  if (mvpOrderedConnectedKeyFrames.empty()) return vector<KeyFrame *>();

  // 获取共视数多与 w 的最后一个迭代器
  vector<int>::iterator it =
      upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w,
                  KeyFrame::weightComp);
  if (it == mvOrderedWeights.end())
    return vector<KeyFrame *>();
  else {
    int n = it - mvOrderedWeights.begin();
    // 返回前 n 个关键帧，这些关键帧共视数大于 w
    return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(),
                              mvpOrderedConnectedKeyFrames.begin() + n);
  }
}

/// @brief 获取关键帧共视数
/// @param pKF 关键帧
/// @return
int KeyFrame::GetWeight(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexConnections);
  if (mConnectedKeyFrameWeights.count(pKF))
    return mConnectedKeyFrameWeights[pKF];
  else
    return 0;
}

/// @brief 根据 MP 观测更新共视关系
void KeyFrame::UpdateConnections() {
  map<KeyFrame *, int> KFcounter;  // 共视计数器  <关键帧, 与当前关键帧的共视数>

  vector<MapPoint *> vpMP;  // 关联到关键点的地图点

  {
    unique_lock<mutex> lockMPs(mMutexFeatures);
    vpMP = mvpMapPoints;
  }

  // For all map points in keyframe check in which other keyframes are they seen
  // Increase counter for those keyframes

  /* 1.遍历观测 MP, 遍历 MP 关联 KF, 统计共视数 */

  // 遍历与关键点关联的地图点
  for (vector<MapPoint *>::iterator vit = vpMP.begin(), vend = vpMP.end();
       vit != vend; vit++) {
    MapPoint *pMP = *vit;

    // 若无关联地图点，跳过
    if (!pMP) continue;

    // 若地图点为坏点，跳过
    if (pMP->isBad()) continue;

    map<KeyFrame *, size_t> observations =
        pMP->GetObservations();  // 观测到地图点的观测关键帧集 <关键帧,
                                 // 地图点对应关键点序号>

    // 遍历关键帧，这些关键帧与当前关键帧有共视关系
    for (map<KeyFrame *, size_t>::iterator mit = observations.begin(),
                                           mend = observations.end();
         mit != mend; mit++) {
      if (mit->first->mnId == mnId) continue;
      KFcounter[mit->first]++;  // 与这些关键帧的共视数 +1
    }
  }

  // This should not happen
  // 判空，应该不会为空
  if (KFcounter.empty()) return;

  /* 2.筛选共视关系，最小共视阈值为 15，若没有一个超过阈值，保留最大共视的
   * KF，同时更新对方共视 */

  // If the counter is greater than threshold add connection
  // In case no keyframe counter is over threshold add the one with maximum
  // counter
  // 如果共视数超过阈值，则添加连接关系。如果没有一个超过阈值，则选择共视数最大的一个
  int nmax = 0;             // 最大共视数
  KeyFrame *pKFmax = NULL;  // 最大共视关键帧
  int th = 15;              // 共视数阈值，常数

  // 这里写为 pair<int, KeyFrame*>
  // 是为了方便直接调用排序算法，不需要定义额外仿函数
  vector<pair<int, KeyFrame *> > vPairs;  // 筛选共视关键帧  <共视数， 关键帧>
  vPairs.reserve(KFcounter.size());
  for (map<KeyFrame *, int>::iterator mit = KFcounter.begin(),
                                      mend = KFcounter.end();
       mit != mend; mit++) {
    // 保存最大共视数的关键帧
    if (mit->second > nmax) {
      nmax = mit->second;
      pKFmax = mit->first;
    }
    // 将共视数超过阈值的关键帧加入向量
    if (mit->second >= th) {
      vPairs.push_back(make_pair(mit->second, mit->first));  // 加入向量
      (mit->first)->AddConnection(this, mit->second);  // 更新对方的共视数据
    }
  }

  // 若为空，则加入共视数最大的关键帧
  if (vPairs.empty()) {
    vPairs.push_back(make_pair(nmax, pKFmax));
    pKFmax->AddConnection(this, nmax);
  }

  /* 3.对共视数排序，相当于调用 UpdateBestCovisibles；填充生成树 */
  sort(vPairs.begin(), vPairs.end());  // 按共视关系排序  小到大
  list<KeyFrame *> lKFs;               // 关键帧列表
  list<int> lWs;                       // 共视数列表
  for (size_t i = 0; i < vPairs.size(); i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  // 填充成员变量
  {
    unique_lock<mutex> lockCon(mMutexConnections);

    // mspConnectedKeyFrames = spConnectedKeyFrames;
    mConnectedKeyFrameWeights = KFcounter;
    mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

    // 如果是第一次被连接，则额外设置生成树信息
    if (mbFirstConnection && mnId != 0) {
      mpParent = mvpOrderedConnectedKeyFrames
                     .front();  // 共视数最多的关键帧作为父关键帧
      mpParent->AddChild(this);  // 更新父关键帧信息
      mbFirstConnection = false;
    }
  }
}

///////////////////////////////////////////////////////////////////
///地图点观测相关函数

/// @brief 添加地图点
/// @param pMP 地图点
/// @param idx 对应关键点序号
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx) {
  unique_lock<mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = pMP;
}

/// @brief 按索引清除地图点匹配
/// @param idx 关键点序号
void KeyFrame::EraseMapPointMatch(const size_t &idx) {
  unique_lock<mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);  // 清除关联, 赋 nullptr
}

/// @brief 按指针删除地图点匹配
/// @param pMP 地图点指针
void KeyFrame::EraseMapPointMatch(MapPoint *pMP) {
  int idx =
      pMP->GetIndexInKeyFrame(this);  // 查询地图点在此帧内对应关键点的序号
  if (idx >= 0)
    mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);  // 清除关联, 赋 nullptr
}

/// @brief 按索引替代地图点匹配
/// @param idx 关键点序号
/// @param pMP 新地图点指针
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP) {
  mvpMapPoints[idx] = pMP;
}

/// @brief 获取地图点
/// @return
set<MapPoint *> KeyFrame::GetMapPoints() {
  unique_lock<mutex> lock(mMutexFeatures);
  set<MapPoint *> s;  // 地图点集合
  // 遍历每个关键点对应的地图点
  for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++) {
    // 判断是否匹配到地图点
    if (!mvpMapPoints[i]) continue;
    MapPoint *pMP = mvpMapPoints[i];
    // 判断这个地图点是否为坏点
    if (!pMP->isBad()) s.insert(pMP);
  }
  return s;
}

/// @brief 追踪到地图点的数量
/// @param minObs 地图点最小观测次数阈值
/// @return
int KeyFrame::TrackedMapPoints(const int &minObs) {
  unique_lock<mutex> lock(mMutexFeatures);

  int nPoints = 0;                    // 地图点计数
  const bool bCheckObs = minObs > 0;  // 标志位，是否使用阈值
  // 遍历关键点对应的地图点
  for (int i = 0; i < N; i++) {
    MapPoint *pMP = mvpMapPoints[i];
    // 如果存在地图点匹配
    if (pMP) {
      // 如果地图点不为坏点
      if (!pMP->isBad()) {
        // 如果需要大于某最小观测次数阈值
        if (bCheckObs) {
          // 判断是否超过阈值
          if (mvpMapPoints[i]->Observations() >= minObs) nPoints++;
        } else
          nPoints++;
      }
    }
  }

  return nPoints;
}

/// @brief 获取匹配地图点集 按关键点索引
/// @return
vector<MapPoint *> KeyFrame::GetMapPointMatches() {
  unique_lock<mutex> lock(mMutexFeatures);
  return mvpMapPoints;
}

/// @brief 获取与关键点关联的地图点
/// @param idx 关键点序号
/// @return
MapPoint *KeyFrame::GetMapPoint(const size_t &idx) {
  unique_lock<mutex> lock(mMutexFeatures);
  return mvpMapPoints[idx];
}

///////////////////////////////////////////////////////////////////
///生成树与回环相关函数

/// @brief 添加子节点
/// @param pKF 子关键帧
void KeyFrame::AddChild(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  mspChildrens.insert(pKF);  // 向 set 插入元素
}

/// @brief 删除子节点
/// @param pKF 子关键帧
void KeyFrame::EraseChild(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  mspChildrens.erase(pKF);  // 从 set 删除元素
}

/// @brief 更改父节点, 添加当前关键帧到父关键帧子节点集中
/// @param pKF 父关键帧
void KeyFrame::ChangeParent(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  mpParent = pKF;  // 更改当前关键帧父节点
  pKF->AddChild(this);  // 将当前关键帧添加到父关键帧的子节点集合中
}

/// @brief 获取子节点集合
/// @return
set<KeyFrame *> KeyFrame::GetChilds() {
  unique_lock<mutex> lockCon(mMutexConnections);
  return mspChildrens;
}

/// @brief 获取父节点
/// @return
KeyFrame *KeyFrame::GetParent() {
  unique_lock<mutex> lockCon(mMutexConnections);
  return mpParent;
}

/// @brief 子节点集合是否包含此关键帧
/// @param pKF 被查询关键帧
/// @return
bool KeyFrame::hasChild(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  return mspChildrens.count(pKF);
}

/// @brief 添加回环边
/// @param pKF 回环连接到的关键帧
void KeyFrame::AddLoopEdge(KeyFrame *pKF) {
  unique_lock<mutex> lockCon(mMutexConnections);
  mbNotErase = true;
  mspLoopEdges.insert(pKF);
}

/// @brief 获取回环边集合
/// @return
set<KeyFrame *> KeyFrame::GetLoopEdges() {
  unique_lock<mutex> lockCon(mMutexConnections);
  return mspLoopEdges;
}

///////////////////////////////////////////////////////////////////
///标志位设置相关函数

/// @brief 设置为不可清除  LoopClosing 中使用，避免 KF 在回环检测中被删除
void KeyFrame::SetNotErase() {
  unique_lock<mutex> lock(mMutexConnections);
  mbNotErase = true;
}

/// @brief 设置为可清除  LoopClosing 中完成回环检测后使用
void KeyFrame::SetErase() {
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (mspLoopEdges.empty()) {
      mbNotErase = false;
    }
  }

  if (mbToBeErased) {
    SetBadFlag();
  }
}

/// @brief 设置为坏节点
void KeyFrame::SetBadFlag() {
  {
    unique_lock<mutex> lock(mMutexConnections);
    if (mnId == 0)
      return;
    else if (mbNotErase) {
      mbToBeErased = true;
      return;
    }
  }

  /* 1.删除共视关键帧中当前帧的信息 */
  for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                                      mend = mConnectedKeyFrameWeights.end();
       mit != mend; mit++)
    mit->first->EraseConnection(this);

  /* 2.删除观测地图点中当前帧的观测信息 */
  for (size_t i = 0; i < mvpMapPoints.size(); i++)
    if (mvpMapPoints[i]) mvpMapPoints[i]->EraseObservation(this);

  /* 3.删除生成树中当前帧的观测信息，更新生成树 */
  {
    unique_lock<mutex> lock(mMutexConnections);
    unique_lock<mutex> lock1(mMutexFeatures);

    mConnectedKeyFrameWeights.clear();
    mvpOrderedConnectedKeyFrames.clear();

    // Update Spanning Tree
    // 更新生成树
    set<KeyFrame *> sParentCandidates;  // 候选父节点，最初只有当前 KF 的父 KF
    sParentCandidates.insert(mpParent);

    // Assign at each iteration one children with a parent (the pair with
    // highest covisibility weight) Include that children as new parent
    // candidate for the rest
    // 为每个子节点赋一个父节点，包括某个子节点是其他子节点的父节点的情况

    // 当子节点不为空时，每次循环寻找一对最佳父子关系
    while (!mspChildrens.empty()) {
      bool bContinue = false;

      int max = -1;  // 最大共视数
      KeyFrame *pC;  // 最佳子关键帧
      KeyFrame *pP;  // 最佳父关键帧

      // 遍历所有父节点与子节点，记录一组最佳共视关系的父、子节点

      // 遍历子关键帧节点
      for (set<KeyFrame *>::iterator sit = mspChildrens.begin(),
                                     send = mspChildrens.end();
           sit != send; sit++) {
        KeyFrame *pKF = *sit;  // 当前子关键帧
        // 坏点忽略
        if (pKF->isBad()) continue;

        // Check if a parent candidate is connected to the keyframe
        // 检查是否有候选父节点与这个关键帧有共视关系
        // 遍历子节点共视关键帧中同时为候选父节点的 KF，记录最佳共视与序号
        vector<KeyFrame *> vpConnected =
            pKF->GetVectorCovisibleKeyFrames();  // 有序共视关键帧向量
        // 遍历子节点共视关键帧，寻找其中位于候选父节点集的 KF，记录最佳共视关系
        for (size_t i = 0, iend = vpConnected.size(); i < iend; i++) {
          // 遍历候选父节点
          for (set<KeyFrame *>::iterator spcit = sParentCandidates.begin(),
                                         spcend = sParentCandidates.end();
               spcit != spcend; spcit++) {
            // 如果子节点共视关键帧与候选父节点相同
            if (vpConnected[i]->mnId == (*spcit)->mnId) {
              int w = pKF->GetWeight(vpConnected[i]);
              if (w > max) {
                pC = pKF;             // 当前帧
                pP = vpConnected[i];  // 最佳共视帧
                max = w;              // 最佳共视关系
                bContinue = true;     // 如果共视关系存在，继续计算
              }
            }
          }
        }
      }

      if (bContinue) {
        pC->ChangeParent(pP);  // 更改父节点，添加到父节点的子节点向量
        sParentCandidates.insert(pC);  // 将这个子节点设置为父节点候选列表
        mspChildrens.erase(pC);  // 从子节点列表中清除这个子节点
      } else
        break;
    }

    // If a children has no covisibility links with any parent candidate, assign
    // to the original parent of this KF
    // 如果剩余子节点没有任何与候选父节点的共视关系，则将被删除节点的父节点设置为他们的父节点
    if (!mspChildrens.empty())
      for (set<KeyFrame *>::iterator sit = mspChildrens.begin();
           sit != mspChildrens.end(); sit++) {
        (*sit)->ChangeParent(mpParent);
      }

    mpParent->EraseChild(this);  // 更新父节点信息
    mTcp = Tcw * mpParent->GetPoseInverse();
    mbBad = true;  // 坏点
  }

  /* 4.从地图、KF 数据库中删除 */
  mpMap->EraseKeyFrame(this);  // 从地图中删除
  mpKeyFrameDB->erase(this);   // 从关键帧数据库中删除
}

/// @brief 是否为坏关键帧
/// @return
bool KeyFrame::isBad() {
  unique_lock<mutex> lock(mMutexConnections);
  return mbBad;
}

/////////////////////////////////////////////////////////////////// 其他函数

/// @brief 获取方形区域中的特征   类似于 Frame
/// @param x 中心 u
/// @param y 中心 v
/// @param r 一半宽
/// @return
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y,
                                           const float &r) const {
  vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX =
      max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (nMinCellX >= mnGridCols) return vIndices;

  const int nMaxCellX = min(
      (int)mnGridCols - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (nMaxCellX < 0) return vIndices;

  const int nMinCellY =
      max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= mnGridRows) return vIndices;

  const int nMaxCellY =
      min((int)mnGridRows - 1,
          (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0) return vIndices;

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const vector<size_t> vCell = mGrid[ix][iy];
      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        // 这里是方形范围
        if (fabs(distx) < r && fabs(disty) < r) vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

/// @brief 点否在图像去畸变范围中
/// @param x
/// @param y
/// @return
bool KeyFrame::IsInImage(const float &x, const float &y) const {
  return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

/// @brief 计算特征点空间位置
/// @param i 特征点索引
/// @return 三维位置向量
cv::Mat KeyFrame::UnprojectStereo(int i) {
  const float z = mvDepth[i];
  if (z > 0) {
    const float u = mvKeys[i].pt.x;
    const float v = mvKeys[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

    unique_lock<mutex> lock(mMutexPose);
    return Twc.rowRange(0, 3).colRange(0, 3) * x3Dc + Twc.rowRange(0, 3).col(3);
  } else
    return cv::Mat();
}

/// @brief 计算场景 1/q 位深度 仅在单目使用
/// @param q 比例，输入 2 时输出深度中位数
/// @return
float KeyFrame::ComputeSceneMedianDepth(const int q) {
  vector<MapPoint *> vpMapPoints;  // 关联到关键点的地图点
  cv::Mat Tcw_;                    // 相机位姿
  {
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPose);
    vpMapPoints = mvpMapPoints;
    Tcw_ = Tcw.clone();
  }

  vector<float> vDepths;  // 深度
  vDepths.reserve(N);
  cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
  Rcw2 = Rcw2.t();
  float zcw = Tcw_.at<float>(2, 3);
  // 遍历特征点对应的地图点
  for (int i = 0; i < N; i++) {
    if (mvpMapPoints[i]) {
      MapPoint *pMP = mvpMapPoints[i];
      cv::Mat x3Dw = pMP->GetWorldPos();  // 特征点空间位置
      float z =
          Rcw2.dot(x3Dw) +
          zcw;  // 相对于相机深度，在单目中不能直接获取深度，所以这里用了坐标变幻
      vDepths.push_back(z);
    }
  }

  sort(vDepths.begin(), vDepths.end());  // 为深度排序

  return vDepths[(vDepths.size() - 1) / q];  // 取 1/q 位数
}

/////////////////////////////////////////////////////////////////
void KeyFrame::UpdateEigenPose() {
  unique_lock<mutex> lock(mMutexPose);
  Eigen::Isometry3d Tcw_eigen;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      Tcw_eigen(i, j) = Tcw.at<float>(i, j);
    }
  }
  tcw_ = Tcw_eigen.translation();
  qcw_ = Eigen::Quaterniond(Tcw_eigen.rotation());
  // qcw_.normalize();
}

void KeyFrame::UpdatePoseFromEigen() {
  unique_lock<mutex> lock(mMutexPose);
  Eigen::Isometry3d Tcw_eigen(qcw_);
  Tcw_eigen.pretranslate(tcw_);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      Tcw.at<float>(i, j) = Tcw_eigen(i, j);
    }
  }
}

}  // namespace ORB_SLAM2
