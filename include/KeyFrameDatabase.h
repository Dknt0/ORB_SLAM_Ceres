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

#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <list>
#include <mutex>
#include <set>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "ORBVocabulary.h"

namespace ORB_SLAM2 {

class KeyFrame;
class Frame;

/// @brief 关键帧数据库
class KeyFrameDatabase {
 public:
  KeyFrameDatabase(const ORBVocabulary& voc);

  void add(KeyFrame* pKF);

  void erase(KeyFrame* pKF);

  void clear();

  // Loop Detection

  std::vector<KeyFrame*> DetectLoopCandidates(KeyFrame* pKF, float minScore);

  // Relocalization

  std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);

 protected:
  // Associated vocabulary
  const ORBVocabulary* mpVoc;  // 视觉字典

  // Inverted file
  // 文件记录不同单词。如果构造一个集合，记录相同单词在哪些文件中出现过，就是逆文件
  std::vector<list<KeyFrame*> >
      mvInvertedFile;  // 逆文件向量  包含单词的关键帧列表
                       // [单词序号]list<关键帧>

  // Mutex
  std::mutex mMutex;  // 关键帧数据库锁
};

}  // namespace ORB_SLAM2

#endif
