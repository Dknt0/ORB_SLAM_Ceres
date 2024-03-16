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

#ifndef MAP_H
#define MAP_H

#include <mutex>
#include <set>

#include "KeyFrame.h"
#include "MapPoint.h"

namespace ORB_SLAM2 {

class MapPoint;
class KeyFrame;

class Map {
 public:
  Map();
  // 增删改

  void AddKeyFrame(KeyFrame* pKF);
  void AddMapPoint(MapPoint* pMP);
  void EraseMapPoint(MapPoint* pMP);
  void EraseKeyFrame(KeyFrame* pKF);
  void SetReferenceMapPoints(const std::vector<MapPoint*>& vpMPs);
  void InformNewBigChange();
  void clear();
  // 查

  std::vector<KeyFrame*> GetAllKeyFrames();
  std::vector<MapPoint*> GetAllMapPoints();
  std::vector<MapPoint*> GetReferenceMapPoints();
  int GetLastBigChangeIdx();
  long unsigned int MapPointsInMap();
  long unsigned KeyFramesInMap();
  long unsigned int GetMaxKFid();

  std::vector<KeyFrame*>
      mvpKeyFrameOrigins;  // 关键帧原点  为啥是个向量，不应该只有一个吗?

  std::mutex mMutexMapUpdate;  // 地图更新锁

  // This avoid that two points are created simultaneously in separate threads
  // (id conflict)
  std::mutex mMutexPointCreation;  // 地图点创建锁

 protected:
  std::set<MapPoint*> mspMapPoints;              // 地图点集合
  std::set<KeyFrame*> mspKeyFrames;              // 关键帧集合
  std::vector<MapPoint*> mvpReferenceMapPoints;  // 参考地图点集 LocalMap MP

  long unsigned int mnMaxKFid;  // 最大关键帧序号

  // Index related to a big change in the map (loop closure, global BA)
  int mnBigChangeIdx;  // 上一次大改变序号

  std::mutex mMutexMap;  // 地图锁
};

}  // namespace ORB_SLAM2

#endif  // MAP_H
