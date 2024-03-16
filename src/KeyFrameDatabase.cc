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

#include "KeyFrameDatabase.h"

#include <mutex>

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

using namespace std;

namespace ORB_SLAM2 {

/// @brief 关键帧数据库构造函数
/// @param voc 词袋
KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary& voc) : mpVoc(&voc) {
  mvInvertedFile.resize(voc.size());  // 设置向量元素数与字典中词汇数相同
}

/// @brief 添加关键帧
/// @param pKF 关键帧
void KeyFrameDatabase::add(KeyFrame* pKF) {
  unique_lock<mutex> lock(mMutex);

  // 遍历这个关键帧的视觉词汇向量，填充逆文件向量
  for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(),
                                        vend = pKF->mBowVec.end();
       vit != vend; vit++)
    mvInvertedFile[vit->first].push_back(pKF);  // 在对应单词位置加入关键帧
}

/// @brief 清除关键帧
/// @param pKF
void KeyFrameDatabase::erase(KeyFrame* pKF) {
  unique_lock<mutex> lock(mMutex);

  // Erase elements in the Inverse File for the entry

  // 遍历这个关键帧的视觉单词向量
  for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(),
                                        vend = pKF->mBowVec.end();
       vit != vend; vit++) {
    // List of keyframes that share the word
    list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];

    // 遍历这个单词列表，删除当前关键帧
    for (list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end();
         lit != lend; lit++) {
      if (pKF == *lit) {
        lKFs.erase(lit);
        break;
      }
    }
  }
}

/// @brief 重置数据库
void KeyFrameDatabase::clear() {
  mvInvertedFile.clear();
  mvInvertedFile.resize(mpVoc->size());
}

/// @brief 检测回环候选关键帧
/// @param pKF 关键帧
/// @param minScore 最小相似性分数
/// @return 候选关键帧向量  无序
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF,
                                                         float minScore) {
  set<KeyFrame*> spConnectedKeyFrames =
      pKF->GetConnectedKeyFrames();  // 共视关键帧集合
  list<KeyFrame*> lKFsSharingWords;  // 有当前 KF 有相同词汇的候选 KF

  // Search all keyframes that share a word with current keyframe
  // Discard keyframes connected to the query keyframe
  /* 1.选择与当前 KF 有相同视觉词汇且不构成共视关系的，作为候选 KF  填充
   * lKFsSharingWords */

  {
    unique_lock<mutex> lock(mMutex);

    // 遍历被搜索 KF 词袋，通过单词从逆文件中获取包含相同单词的候选 KF
    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(),
                                          vend = pKF->mBowVec.end();
         vit != vend; vit++) {
      list<KeyFrame*>& lKFs =
          mvInvertedFile[vit->first];  // 包含这个单词的关键帧列表

      // 遍历逆文件，寻找包含相同词汇的关键帧
      for (list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end();
           lit != lend; lit++) {
        KeyFrame* pKFi = *lit;  // 候选 KF
        // 如果候选 KF 是第一次被查询，重置候选 KF 中回环相关变量
        if (pKFi->mnLoopQuery != pKF->mnId) {
          pKFi->mnLoopWords = 0;
          // 如果不存在共视关系
          if (!spConnectedKeyFrames.count(pKFi)) {
            pKFi->mnLoopQuery = pKF->mnId;  // 设置回环关键帧序号?
            lKFsSharingWords.push_back(pKFi);
          }
        }
        // 与当前 KF 相同单词数 +1
        pKFi->mnLoopWords++;
      }
    }
  }

  // 如果不存在有相同视觉词汇的非共视关键帧，返回空向量
  if (lKFsSharingWords.empty()) return vector<KeyFrame*>();

  /* 2.计算相同单词数阈值，阈值为最多相同单词数的 0.8 倍 */
  list<pair<float, KeyFrame*> >
      lScoreAndMatch;  // 匹配分数-关键帧列表  list<pair<相似性得分, 关键帧>>

  // Only compare against those keyframes that share enough words
  // 只比较有足够相同词汇的关键帧

  int maxCommonWords = 0;  // 最多相同词汇数量
  // 遍历有相同词汇的关键帧，记录最多相同词汇数量
  for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                                 lend = lKFsSharingWords.end();
       lit != lend; lit++) {
    if ((*lit)->mnLoopWords > maxCommonWords)
      maxCommonWords = (*lit)->mnLoopWords;
  }

  int minCommonWords = maxCommonWords * 0.8f;  // 最少相同词汇数量阈值

  int nscores = 0;  // 达标关键帧数量

  // Compute similarity score. Retain the matches whose score is higher than
  // minScore
  /* 3.按照相同单词数阈值对候选关键帧进行筛选，对满足要求的候选 KF
   * 计算相似性得分，保留得分高于阈值的关键帧  填充 lScoreAndMatch */

  // 遍历有相同词汇的关键帧
  for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                                 lend = lKFsSharingWords.end();
       lit != lend; lit++) {
    KeyFrame* pKFi = *lit;  // 候选关键帧
    // 如果相同词汇数大于阈值，计算相似性得分，记录
    if (pKFi->mnLoopWords > minCommonWords) {
      nscores++;

      float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);  // 计算相似性得分

      pKFi->mLoopScore = si;
      // 如果高于最小得分阈值，存入匹配关键帧列表
      if (si >= minScore) lScoreAndMatch.push_back(make_pair(si, pKFi));
    }
  }

  // 如果匹配关键帧列表为空，返回
  if (lScoreAndMatch.empty()) return vector<KeyFrame*>();

  list<pair<float, KeyFrame*> >
      lAccScoreAndMatch;  // 累计相似性得分-关键帧列表 list<pair<累计相似性得分,
                          // 关键帧>>
  float bestAccScore = minScore;

  // Let's now accumulate score by covisibility
  /* 4.依据共视关系计算累计相似性分数，考虑候选 KF 共视关键帧中同样为候选 KF
   * 的那部分，对他们与被搜索帧的相似性得分求和  填充 lAccScoreAndMatch */

  // 遍历匹配分数-关键帧列表
  for (list<pair<float, KeyFrame*> >::iterator it = lScoreAndMatch.begin(),
                                               itend = lScoreAndMatch.end();
       it != itend; it++) {
    KeyFrame* pKFi = it->second;  // 关键帧
    vector<KeyFrame*> vpNeighs =
        pKFi->GetBestCovisibilityKeyFrames(10);  // 获取 10 个最佳共视关键帧

    float bestScore = it->first;
    float accScore = it->first;
    KeyFrame* pBestKF = pKFi;
    // 遍历关键帧的 10 个共视关键帧
    for (vector<KeyFrame*>::iterator vit = vpNeighs.begin(),
                                     vend = vpNeighs.end();
         vit != vend; vit++) {
      KeyFrame* pKF2 = *vit;
      // 如果 pKF2 也是被搜索帧的候选回环关键帧
      if (pKF2->mnLoopQuery == pKF->mnId &&
          pKF2->mnLoopWords > minCommonWords) {
        accScore += pKF2->mLoopScore;  // 累计分数
        // 记录相似性得分最高的关键帧和分数
        if (pKF2->mLoopScore > bestScore) {
          pBestKF = pKF2;  // 注意这里可能将 pBestKF 变为其他关键帧
          bestScore = pKF2->mLoopScore;
        }
      }
    }

    lAccScoreAndMatch.push_back(
        make_pair(accScore, pBestKF));  // 注意 pBestKF 不一定是 pKFi,
                                        // 这个列表中可能有相同的关键帧多次出现
    if (accScore > bestAccScore) bestAccScore = accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  /* 5.保留所有累计得分高于最高得分 0.75 倍的候选关键帧 */
  float minScoreToRetain = 0.75f * bestAccScore;

  set<KeyFrame*> spAlreadyAddedKF;
  vector<KeyFrame*> vpLoopCandidates;  // 回环候选关键帧
  vpLoopCandidates.reserve(lAccScoreAndMatch.size());

  /* 6.筛选候选关键帧，防止重复出现 */

  // 遍历累计匹配分数-关键帧列表
  for (list<pair<float, KeyFrame*> >::iterator it = lAccScoreAndMatch.begin(),
                                               itend = lAccScoreAndMatch.end();
       it != itend; it++) {
    if (it->first > minScoreToRetain) {
      KeyFrame* pKFi = it->second;
      // 防止同一关键帧重复出现
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpLoopCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }

  return vpLoopCandidates;
}

/// @brief 检测重定位候选关键帧
///     Tracking::Relocalization
/// @param F 帧
/// @return 候选关键帧向量
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame* F) {
  // 这个函数和检测回环候选关键帧实现相似

  list<KeyFrame*> lKFsSharingWords;  // 有相同词汇的关键帧

  // Search all keyframes that share a word with current frame
  /* 1.选择与当前 F 有相同视觉词汇的 KF 作为候选 KF  填充 lKFsSharingWords */

  {
    unique_lock<mutex> lock(mMutex);

    // 遍历被搜索关键帧的视觉单词向量
    for (DBoW2::BowVector::const_iterator vit = F->mBowVec.begin(),
                                          vend = F->mBowVec.end();
         vit != vend; vit++) {
      list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];

      // 遍历逆文件列表
      for (list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end();
           lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        // 重置 KeyFrame 变量
        if (pKFi->mnRelocQuery != F->mnId) {
          pKFi->mnRelocWords = 0;
          pKFi->mnRelocQuery = F->mnId;
          lKFsSharingWords.push_back(pKFi);
        }
        pKFi->mnRelocWords++;
      }
    }
  }

  // 如果不存在有相同视觉词汇的关键帧，返回空向量
  if (lKFsSharingWords.empty()) return vector<KeyFrame*>();

  // Only compare against those keyframes that share enough words
  // 只比较有足够相同词汇的关键帧

  int maxCommonWords = 0;  // 最多相同词汇数量
  for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                                 lend = lKFsSharingWords.end();
       lit != lend; lit++) {
    if ((*lit)->mnRelocWords > maxCommonWords)
      maxCommonWords = (*lit)->mnRelocWords;
  }

  int minCommonWords = maxCommonWords * 0.8f;  // 最少相同词汇数量

  /* 2.计算相同单词数阈值，阈值为最多相同单词数的 0.8 倍 */

  list<pair<float, KeyFrame*> >
      lScoreAndMatch;  // 相似性得分-关键帧列表  list<pair<匹配分数, 关键帧>>

  int nscores = 0;  // 达标关键帧数量

  // Compute similarity score.
  /* 3.按照相同单词数阈值对候选关键帧进行筛选，对满足要求的候选 KF
   * 计算相似性得分  填充 lScoreAndMatch */

  // 遍历有相同词汇的关键帧
  for (list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                                 lend = lKFsSharingWords.end();
       lit != lend; lit++) {
    KeyFrame* pKFi = *lit;
    // 如果与被搜索关键帧相同词汇的数量大于最少最多相同词汇数量
    if (pKFi->mnRelocWords > minCommonWords) {
      nscores++;
      float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);  // 计算相似性得分
      pKFi->mRelocScore = si;
      lScoreAndMatch.push_back(make_pair(si, pKFi));
    }
  }

  // 如果匹配关键帧列表为空，返回
  if (lScoreAndMatch.empty()) return vector<KeyFrame*>();

  list<pair<float, KeyFrame*> >
      lAccScoreAndMatch;  // 累计匹配分数-关键帧列表  list<pair<累计匹配分数,
                          // 关键帧>>
  float bestAccScore = 0;

  // Lets now accumulate score by covisibility
  /* 4.依据共视关系计算累计相似性分数，考虑候选 KF 共视关键帧中同样为候选 KF
   * 的那部分，对他们与被搜索帧的相似性得分求和  填充 lAccScoreAndMatch */

  // 遍历匹配分数-关键帧列表
  for (list<pair<float, KeyFrame*> >::iterator it = lScoreAndMatch.begin(),
                                               itend = lScoreAndMatch.end();
       it != itend; it++) {
    KeyFrame* pKFi = it->second;  // 关键帧
    vector<KeyFrame*> vpNeighs =
        pKFi->GetBestCovisibilityKeyFrames(10);  // 获取 10 个最佳共视关键帧

    float bestScore = it->first;
    float accScore = bestScore;
    KeyFrame* pBestKF = pKFi;
    // 遍历关键帧的 10 个共视关键帧
    for (vector<KeyFrame*>::iterator vit = vpNeighs.begin(),
                                     vend = vpNeighs.end();
         vit != vend; vit++) {
      KeyFrame* pKF2 = *vit;
      // 如果 pKF2 不是被搜索帧的候选重定位关键帧，忽略
      if (pKF2->mnRelocQuery != F->mnId) continue;

      accScore += pKF2->mRelocScore;
      if (pKF2->mRelocScore > bestScore) {
        pBestKF = pKF2;  // 注意这里可能将 pBestKF 变为其他关键帧
        bestScore = pKF2->mRelocScore;
      }
    }
    lAccScoreAndMatch.push_back(
        make_pair(accScore, pBestKF));  // 注意 pBestKF 不一定是 pKFi,
                                        // 这个列表中可能有相同的关键帧多次出现
    if (accScore > bestAccScore) bestAccScore = accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  /* 5.保留所有累计得分高于最高得分 0.75 倍的候选关键帧 */
  float minScoreToRetain = 0.75f * bestAccScore;

  set<KeyFrame*> spAlreadyAddedKF;
  vector<KeyFrame*> vpRelocCandidates;
  vpRelocCandidates.reserve(lAccScoreAndMatch.size());

  /* 6.筛选候选关键帧，防止重复出现 */

  // 遍历累计匹配分数-关键帧列表
  for (list<pair<float, KeyFrame*> >::iterator it = lAccScoreAndMatch.begin(),
                                               itend = lAccScoreAndMatch.end();
       it != itend; it++) {
    const float& si = it->first;
    if (si > minScoreToRetain) {
      KeyFrame* pKFi = it->second;
      // 防止同一关键帧重复出现
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpRelocCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }

  return vpRelocCandidates;
}

}  // namespace ORB_SLAM2
