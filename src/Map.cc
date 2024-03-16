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

#include "Map.h"

#include <mutex>

namespace ORB_SLAM2 {

/// @brief 构造函数
Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0) {}

/// @brief 添加关键帧
/// @param pKF
void Map::AddKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexMap);
  mspKeyFrames.insert(pKF);
  if (pKF->mnId > mnMaxKFid) mnMaxKFid = pKF->mnId;
}

/// @brief 添加地图点
/// @param pMP
void Map::AddMapPoint(MapPoint *pMP) {
  unique_lock<mutex> lock(mMutexMap);
  mspMapPoints.insert(pMP);
}

/// @brief 删除地图点  没有从内存中删除点！
/// @param pMP
void Map::EraseMapPoint(MapPoint *pMP) {
  unique_lock<mutex> lock(mMutexMap);
  mspMapPoints.erase(pMP);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

/// @brief 删除关键帧  没有从内存中删除帧！
/// @param pKF
void Map::EraseKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexMap);
  mspKeyFrames.erase(pKF);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

/// @brief 设置参考地图点
/// @param vpMPs
void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs) {
  unique_lock<mutex> lock(mMutexMap);
  mvpReferenceMapPoints = vpMPs;
}

/// @brief 通知新大改动
void Map::InformNewBigChange() {
  unique_lock<mutex> lock(mMutexMap);
  mnBigChangeIdx++;
}

/// @brief 获取上一次大改动序号
/// @return
int Map::GetLastBigChangeIdx() {
  unique_lock<mutex> lock(mMutexMap);
  return mnBigChangeIdx;
}

/// @brief 获取全部关键帧
/// @return
vector<KeyFrame *> Map::GetAllKeyFrames() {
  unique_lock<mutex> lock(mMutexMap);
  return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
}

/// @brief 获取全部地图点
/// @return
vector<MapPoint *> Map::GetAllMapPoints() {
  unique_lock<mutex> lock(mMutexMap);
  return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
}

/// @brief 获取地图点总数
/// @return
long unsigned int Map::MapPointsInMap() {
  unique_lock<mutex> lock(mMutexMap);
  return mspMapPoints.size();
}

/// @brief 获取关键帧总数
/// @return
long unsigned int Map::KeyFramesInMap() {
  unique_lock<mutex> lock(mMutexMap);
  return mspKeyFrames.size();
}

/// @brief 获取参考地图点
/// @return
vector<MapPoint *> Map::GetReferenceMapPoints() {
  unique_lock<mutex> lock(mMutexMap);
  return mvpReferenceMapPoints;
}

/// @brief 获取最大关键帧 ID
/// @return
long unsigned int Map::GetMaxKFid() {
  unique_lock<mutex> lock(mMutexMap);
  return mnMaxKFid;
}

/// @brief 清除所有数据
void Map::clear() {
  // 这里清理了堆区内存
  for (set<MapPoint *>::iterator sit = mspMapPoints.begin(),
                                 send = mspMapPoints.end();
       sit != send; sit++)
    delete *sit;

  for (set<KeyFrame *>::iterator sit = mspKeyFrames.begin(),
                                 send = mspKeyFrames.end();
       sit != send; sit++)
    delete *sit;

  mspMapPoints.clear();
  mspKeyFrames.clear();
  mnMaxKFid = 0;
  mvpReferenceMapPoints.clear();
  mvpKeyFrameOrigins.clear();
}

}  // namespace ORB_SLAM2
