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

#include "LoopClosing.h"

#include <unistd.h>

#include <mutex>
#include <thread>

#include "CeresOptimizer.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "Sim3Solver.h"

namespace ORB_SLAM2 {

/// @brief 构造函数
/// @param pMap
/// @param pDB
/// @param pVoc
/// @param bFixScale
LoopClosing::LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,
                         const bool bFixScale)
    : mpMap(pMap),
      mpKeyFrameDB(pDB),
      mpORBVocabulary(pVoc),
      mpMatchedKF(NULL),
      mLastLoopKFid(0),
      mbResetRequested(false),
      mbFinishRequested(false),
      mbFinished(true),
      mbRunningGBA(false),
      mbFinishedGBA(true),
      mbStopGBA(false),
      mpThreadGBA(NULL),
      mbFixScale(bFixScale),
      mnFullBAIdx(0) {
  mnCovisibilityConsistencyTh = 3;
}

/// @brief 设置 Tracking 指针
/// @param pTracker
void LoopClosing::SetTracker(Tracking* pTracker) { mpTracker = pTracker; }

/// @brief 设置 LocalMapping 指针
/// @param pLocalMapper
void LoopClosing::SetLocalMapper(LocalMapping* pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

/// @brief 运行  线程函数
void LoopClosing::Run() {
  mbFinished = false;

  while (1) {
    // 检测是否存在新 KF
    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      // 检测回环候选 KF 并检查共视一致性
      // Detect loop candidates and check covisibility consistency
      if (DetectLoop()) {
        // 如果检测出回环，求解 Sim3
        // Compute similarity transformation [sR|t]
        // In the stereo/RGBD case s=1
        if (ComputeSim3()) {
          // 回环修正
          // Perform loop fusion and pose graph optimization
          CorrectLoop();
        }
      }
    }

    // 重置检查
    ResetIfRequested();

    // 终止检查
    if (CheckFinish()) break;

    // 延时 5000 us
    usleep(5000);
  }

  // 设置终止
  SetFinish();
}

/// @brief 添加关键帧
/// @param pKF
void LoopClosing::InsertKeyFrame(KeyFrame* pKF) {
  unique_lock<mutex> lock(mMutexLoopQueue);
  if (pKF->mnId != 0) mlpLoopKeyFrameQueue.push_back(pKF);
}

/// @brief KF 队列中是否有 KF
/// @return
bool LoopClosing::CheckNewKeyFrames() {
  unique_lock<mutex> lock(mMutexLoopQueue);
  return (!mlpLoopKeyFrameQueue.empty());
}

/// @brief 闭环检测
/// @return
bool LoopClosing::DetectLoop() {
  /* 从队列中获取 KF */
  {
    unique_lock<mutex> lock(mMutexLoopQueue);
    mpCurrentKF = mlpLoopKeyFrameQueue.front();
    mlpLoopKeyFrameQueue.pop_front();
    // Avoid that a keyframe can be erased while it is being process by this
    // thread 设置当前 KF 不可清除，避免当前 KF 被 LocalMapping 剔除
    mpCurrentKF->SetNotErase();
  }

  /* 如果距离上次回环检测不超过 10 KF，忽略 */
  // If the map contains less than 10 KF or less than 10 KF have passed from
  // last loop detection
  if (mpCurrentKF->mnId < mLastLoopKFid + 10) {
    // 添加到 KFDB
    mpKeyFrameDB->add(mpCurrentKF);
    // 设置当前 KF 可清除
    mpCurrentKF->SetErase();
    return false;
  }

  /* 从当前 KF 共视 KF 中计算最小参考词袋相似性得分 */
  // 注意，共视数和词袋相似性得分是两个概念，通常是正相关的
  // Compute reference BoW similarity score
  // This is the lowest score to a connected keyframe in the covisibility graph
  // We will impose loop candidates to have a higher similarity than this
  const vector<KeyFrame*> vpConnectedKeyFrames =
      mpCurrentKF
          ->GetVectorCovisibleKeyFrames();  // 有序共视关键帧向量  共视大于 15
  const DBoW2::BowVector& CurrentBowVec =
      mpCurrentKF->mBowVec;  // 当前 KF 视觉描述向量
  float minScore = 1;  // 最小词袋相似性得分得分，用于筛选候选 KF
  // 遍历共视 KF，寻找最小相似性得分
  for (size_t i = 0; i < vpConnectedKeyFrames.size(); i++) {
    KeyFrame* pKF = vpConnectedKeyFrames[i];
    if (pKF->isBad()) continue;
    const DBoW2::BowVector& BowVec = pKF->mBowVec;

    float score = mpORBVocabulary->score(CurrentBowVec, BowVec);  // 相似性得分

    if (score < minScore) minScore = score;
  }

  /* 检测相似性得分超过最小值的回环候选 KF */
  // Query the database imposing the minimum score
  vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(
      mpCurrentKF, minScore);  // 回环候选 KF 集

  /* 如果没有候选 KF，清空历史共视组集，返回 */
  // If there are no loop candidates, just add new keyframe and return false
  if (vpCandidateKFs.empty()) {
    mpKeyFrameDB->add(mpCurrentKF);
    // 注意这里清空了共视组集
    mvConsistentGroups.clear();
    mpCurrentKF->SetErase();
    return false;
  }

  /* 共视组一致性检测 */
  // 检查回环候选 KF 与之前候选 KF 的一致性
  // 每一个候选 KF 扩张为一个共视组，包含候选 KF 的共视 KF
  // 如果两个共视组包含至少一个相同 KF，则认为他们具有一致性
  // 仅当在几帧候选 KF 共视组检测出一致性时，才接受回环结果
  // For each loop candidate check consistency with previous loop candidates
  // Each candidate expands a covisibility group (keyframes connected to the
  // loop candidate in the covisibility graph) A group is consistent with a
  // previous group if they share at least a keyframe We must detect a
  // consistent loop in several consecutive keyframes to accept it
  mvpEnoughConsistentCandidates.clear();

  // 一个历史共视组只能与一个候选 KF 共视组构成一致性关系
  vector<ConsistentGroup>
      vCurrentConsistentGroups;  //  当前共视组集   当前 KF 的候选 KF 共视组集
  vector<bool> vbConsistentGroup(
      mvConsistentGroups.size(),
      false);  // 历史共视组是否满足一致性标志位  按照历史共视组顺序索引
  // 遍历回环候选 KF 集
  for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++) {
    KeyFrame* pCandidateKF = vpCandidateKFs[i];  // 候选 KFi

    set<KeyFrame*> spCandidateGroup =
        pCandidateKF->GetConnectedKeyFrames();  // 候选 KFi 共视 KF 集
    spCandidateGroup.insert(pCandidateKF);

    bool bEnoughConsistent = false;        // 足够一致性
    bool bConsistentForSomeGroup = false;  // 与某些组一致
    // 遍历历史共视组集
    for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++) {
      set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;  // 共视组 i

      bool bConsistent = false;  // 候选 KFi 与当前共视组 一致性标志位
      // 当前候选 KF 与共视组一致性检测，即当前候选 KF
      // 共视组与之前的共视组存在一致关系
      // 遍历候选 KF 共视组中的 KF
      for (set<KeyFrame*>::iterator sit = spCandidateGroup.begin(),
                                    send = spCandidateGroup.end();
           sit != send; sit++) {
        // 在共视组 i 中查询当前候选 KFi 共视组中的 KF
        if (sPreviousGroup.count(*sit)) {
          // 只要有一个 KF 相同，就具备一致性
          bConsistent = true;
          bConsistentForSomeGroup = true;
          break;
        }
      }

      // 候选 KFi 与当前共视组一致性成立
      if (bConsistent) {
        int nPreviousConsistency =
            mvConsistentGroups[iG].second;  // 历史共视组一致性 即一致共视组数量
        int nCurrentConsistency =
            nPreviousConsistency + 1;  // 当前候选 KF 共视组一致性
        // 如果当前历史共视组没有对应到一致的候选 KF
        if (!vbConsistentGroup[iG]) {
          ConsistentGroup cg =
              make_pair(spCandidateGroup,
                        nCurrentConsistency);  // 为当前候选 KF 创建共视组
          vCurrentConsistentGroups.push_back(cg);
          // 保证历史共视组不会与多个候选 KF 共视组构成一致关系
          vbConsistentGroup[iG] =
              true;  // this avoid to include the same group more than once
        }
        // 如果当前候选 KF 共视组一致性大于阈值，记录到成员变量中
        if (nCurrentConsistency >= mnCovisibilityConsistencyTh &&
            !bEnoughConsistent) {
          mvpEnoughConsistentCandidates.push_back(pCandidateKF);
          // 防止重复添加候选 KF
          bEnoughConsistent =
              true;  // this avoid to insert the same candidate more than once
        }
      }
    }

    // 如果当前候选 KF
    // 共视组与历史共视组不构成一致关系，创建共视组，一致共视组数为 0 If the
    // group is not consistent with any previous group insert with consistency
    // counter set to zero
    if (!bConsistentForSomeGroup) {
      ConsistentGroup cg = make_pair(spCandidateGroup, 0);  // 创建共视组
      vCurrentConsistentGroups.push_back(cg);
    }
  }

  // Update Covisibility Consistent Groups
  mvConsistentGroups = vCurrentConsistentGroups;  // 更新共视组

  // Add Current Keyframe to database
  mpKeyFrameDB->add(mpCurrentKF);

  // 如果没有满足一致性阈值的候选 KF，回环检测无效
  if (mvpEnoughConsistentCandidates.empty()) {
    mpCurrentKF->SetErase();
    return false;
  } else {
    // 注意，如果回环检测成功，当前 KF 保持不可清除状态
    return true;
  }

  // 以下代码冗余
  mpCurrentKF->SetErase();
  return false;
}

/// @brief 计算 Sim3 变换
/// @return
bool LoopClosing::ComputeSim3() {
  // 当检测出满足一致性阈值的候选 KF 时，执行此函数，寻找最佳参考 KF，计算当前
  // KF 与参考 KF 相对位姿，为优化提供初值

  // 为每一个候选 KF 计算 Sim3
  // For each consistent loop candidate we try to compute a Sim3
  const int nInitialCandidates =
      mvpEnoughConsistentCandidates.size();  // 初始候选 KF 数量

  // We compute first ORB matches for each candidate
  // If enough matches are found, we setup a Sim3Solver
  ORBmatcher matcher(0.75, true);

  vector<Sim3Solver*> vpSim3Solvers;  // Sim3 求解器
  vpSim3Solvers.resize(nInitialCandidates);

  vector<vector<MapPoint*> >
      vvpMapPointMatches;  // 当前 KF 对候选 KF MP 词袋匹配结果
  vvpMapPointMatches.resize(nInitialCandidates);

  vector<bool> vbDiscarded;  // 候选 KF 剔除标志位
  vbDiscarded.resize(nInitialCandidates);

  int nCandidates = 0;  // 足够匹配数的候选 KF 数量

  /* 词袋匹配当前 KF 与候选 KF，为匹配数量足够的候选 KF 创建 Sim3 求解器 */
  // 遍历初始候选 KF
  for (int i = 0; i < nInitialCandidates; i++) {
    KeyFrame* pKF = mvpEnoughConsistentCandidates[i];  // 候选 KF

    // avoid that local mapping erase it while it is being processed in this
    // thread
    pKF->SetNotErase();

    // 忽略坏 KF
    if (pKF->isBad()) {
      vbDiscarded[i] = true;
      continue;
    }

    // 词袋匹配
    int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF,
                                       vvpMapPointMatches[i]);  // 匹配点对数

    // 如果匹配点对超过 20 个，创建 Sim3 求解器
    if (nmatches < 20) {
      vbDiscarded[i] = true;
      continue;
    } else {
      Sim3Solver* pSolver =
          new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);
      pSolver->SetRansacParameters(0.99, 20, 300);
      vpSim3Solvers[i] = pSolver;
    }

    nCandidates++;
  }

  bool bMatch = false;  // 匹配成功标志位

  /* 循环执行 Sim3Solver RANSAC 迭代，寻找内点数足够的匹配，优化 */
  // 循环，直到寻找到匹配或候选 KF 全部淘汰
  // Perform alternatively RANSAC iterations for each candidate
  // until one is succesful or all fail
  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nInitialCandidates; i++) {
      if (vbDiscarded[i]) continue;

      KeyFrame* pKF = mvpEnoughConsistentCandidates[i];  // 候选 KF

      // Perform 5 Ransac Iterations
      vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;

      // 执行 5 次 RANSAC 迭代
      Sim3Solver* pSolver = vpSim3Solvers[i];
      cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers,
                                     nInliers);  // 候选 KF 相对于当前 KF

      // If Ransac reachs max. iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        nCandidates--;
      }

      // 只要计算出结果，就进行优化
      // If RANSAC returns a Sim3, perform a guided matching and optimize with
      // all correspondences
      if (!Scm.empty()) {
        vector<MapPoint*> vpMapPointMatches(
            vvpMapPointMatches[i].size(),
            static_cast<MapPoint*>(NULL));  // 当前 KF 对候选 KF MP 词袋匹配
        for (size_t j = 0, jend = vbInliers.size(); j < jend; j++) {
          // 如果是内点，添加匹配关系
          if (vbInliers[j]) vpMapPointMatches[j] = vvpMapPointMatches[i][j];
        }

        cv::Mat R = pSolver->GetEstimatedRotation();  // 候选 KF 相对于当前 KF
        cv::Mat t =
            pSolver->GetEstimatedTranslation();  // 候选 KF 相对于当前 KF
        const float s = pSolver->GetEstimatedScale();  // 候选 KF 相对于当前 KF
        /* Sim3 互投影匹配 */
        matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5);

        Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
        // /* Sim3 优化 */
        // const int nInliers = Optimizer::OptimizeSim3(
        //     mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

        const int nInliers = CeresOptimizer::OptimizeSim3(
            mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);
        // std::cout << "g2o res: " << nInliers << " loopclosing test: " << test_nInliers << std::endl;
        // std::cout << "loopclosing test: " << nInliers << std::endl;

        // 如果优化内点数多于 20，计算成功
        // If optimization is succesful stop ransacs and continue
        if (nInliers >= 20) {
          bMatch = true;
          mpMatchedKF = pKF;
          Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),
                    Converter::toVector3d(pKF->GetTranslation()),
                    1.0);         // 匹配 KF 位姿 Smw
          mg2oScw = gScm * gSmw;  // 当前 KF 位姿 Scw
          mScw = Converter::toCvMat(mg2oScw);

          mvpCurrentMatchedPoints =
              vpMapPointMatches;  // 当前 KF 对候选 KF MP 词袋匹配
          break;
        }
      }
    }
  }

  /* 如果没有满足内点要求的匹配 KF，计算失败 */
  if (!bMatch) {
    for (int i = 0; i < nInitialCandidates; i++)
      mvpEnoughConsistentCandidates[i]->SetErase();
    mpCurrentKF->SetErase();
    return false;
  }

  /* 匹配当前 KF 与回环匹配 KF 局部范围内的地图点 */
  // Retrieve MapPoints seen in Loop Keyframe and neighbors
  vector<KeyFrame*> vpLoopConnectedKFs =
      mpMatchedKF->GetVectorCovisibleKeyFrames();  // 匹配 KF 及其近邻 KF
  vpLoopConnectedKFs.push_back(mpMatchedKF);
  mvpLoopMapPoints.clear();
  // 遍历匹配 KF 局部范围内 KF
  for (vector<KeyFrame*>::iterator vit = vpLoopConnectedKFs.begin();
       vit != vpLoopConnectedKFs.end(); vit++) {
    KeyFrame* pKF = *vit;
    vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
    for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
      MapPoint* pMP = vpMapPoints[i];
      if (pMP) {
        // 防止重复添加 MP
        if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId) {
          mvpLoopMapPoints.push_back(pMP);
          pMP->mnLoopPointForKF = mpCurrentKF->mnId;
        }
      }
    }
  }

  // 回环局部地图投影匹配
  // Find more matches projecting with the computed Sim3
  matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints,
                             mvpCurrentMatchedPoints, 10);

  // If enough matches accept Loop
  int nTotalMatches = 0;  // 匹配总数
  for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++) {
    if (mvpCurrentMatchedPoints[i]) nTotalMatches++;
  }

  // 匹配总数多于 40 接受回环
  if (nTotalMatches >= 40) {
    for (int i = 0; i < nInitialCandidates; i++)
      if (mvpEnoughConsistentCandidates[i] != mpMatchedKF)
        mvpEnoughConsistentCandidates[i]->SetErase();
    return true;
  } else {
    for (int i = 0; i < nInitialCandidates; i++)
      mvpEnoughConsistentCandidates[i]->SetErase();
    mpCurrentKF->SetErase();
    return false;
  }
}

/// @brief 闭环校正
void LoopClosing::CorrectLoop() {
  cout << "Loop detected!" << endl;

  /* 向 LocalMapping 发送信号，暂停插入新 KF */
  // Send a stop signal to Local Mapping
  // Avoid new keyframes are inserted while correcting the loop
  mpLocalMapper->RequestStop();

  /* 如果当前有全局 BA 执行，中断之 */
  // If a Global Bundle Adjustment is running, abort it
  if (isRunningGBA()) {
    unique_lock<mutex> lock(mMutexGBA);
    mbStopGBA = true;

    mnFullBAIdx++;

    if (mpThreadGBA) {
      mpThreadGBA->detach();  // 分离线程
      delete mpThreadGBA;
    }
  }

  /* 等待 LocalMapping 完全停止 */
  // Wait until Local Mapping has effectively stopped
  while (!mpLocalMapper->isStopped()) {
    usleep(1000);
  }

  // 更新当前 KF
  // 共视关系，注意，更新在当前地图中进行，这个共视关系不包括同回环局部地图
  // Ensure current keyframe is updated
  mpCurrentKF->UpdateConnections();

  // 获取当前 KF 共视 KF，传播计算 Sim3 位姿
  // Retrive keyframes connected to the current keyframe and compute corrected
  // Sim3 pose by propagation
  mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
  mvpCurrentConnectedKFs.push_back(mpCurrentKF);

  KeyFrameAndPose CorrectedSim3;  // 当前 KF 局部范围内 KF 修正后 Scw 映射
  KeyFrameAndPose NonCorrectedSim3;  // 当前 KF 局部范围内 KF 修正前 Scw 映射
  CorrectedSim3[mpCurrentKF] = mg2oScw;  // 将当前 KF 修正后 Scw 加入映射
  cv::Mat Twc = mpCurrentKF->GetPoseInverse();  // 当前 KF 修正前 Twc

  // 计算并记录当前 KF 局部范围内 KF 与 MP 修正后的坐标
  {
    // Get Map Mutex
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    /* 遍历当前 KF 局部范围内 KF，计算并记录其修正前后的位姿 */
    for (vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(),
                                     vend = mvpCurrentConnectedKFs.end();
         vit != vend; vit++) {
      KeyFrame* pKFi = *vit;

      cv::Mat Tiw = pKFi->GetPose();  // Tiw

      // 对于当前 KF 的共视 KF，计算其与当前 KF 的帧间位姿 Tic
      if (pKFi != mpCurrentKF) {
        cv::Mat Tic = Tiw * Twc;
        cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
        cv::Mat tic = Tic.rowRange(0, 3).col(3);
        Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic),
                    1.0);                         // 修正前 Sic
        Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;  // 修正后 Sic
        // Pose corrected with the Sim3 of the loop closure
        CorrectedSim3[pKFi] = g2oCorrectedSiw;
      }

      cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
      cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
      Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw),
                  1.0);  // 修正前 Sic
      // Pose without correction
      NonCorrectedSim3[pKFi] = g2oSiw;
    }

    /* 遍历当前 KF 局部范围内 KF，修正当前 KF 局部范围内 MP
     * 空间坐标，使其对齐到回环的另一边 */
    // Correct all MapPoints obsrved by current keyframe and neighbors, so that
    // they align with the other side of the loop
    for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(),
                                   mend = CorrectedSim3.end();
         mit != mend; mit++) {
      KeyFrame* pKFi = mit->first;
      Sim3 g2oCorrectedSiw = mit->second;                // 修正后 Siw
      Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();  // 修正后 Swi

      Sim3 g2oSiw = NonCorrectedSim3[pKFi];  // 修正前 Siw

      vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();  // KFi 观测 MP
      // 遍历 KFi 观测 MP
      for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++) {
        MapPoint* pMPi = vpMPsi[iMP];
        if (!pMPi) continue;
        if (pMPi->isBad()) continue;
        // 防止重复修正
        if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId) continue;

        // Project with non-corrected pose and project back with corrected pose
        cv::Mat P3Dw = pMPi->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw =
            Converter::toVector3d(P3Dw);  // 修正前坐标
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw =
            g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));  // 修正后坐标

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        // 注意，这里直接改变了 MP 坐标
        pMPi->SetWorldPos(cvCorrectedP3Dw);
        pMPi->mnCorrectedByKF = mpCurrentKF->mnId;  // 记录
        pMPi->mnCorrectedReference = pKFi->mnId;
        pMPi->mnCorrectedReferencePtr = pKFi;
        pMPi->UpdateNormalAndDepth();
      }

      // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3
      // (scale translation)
      Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
      Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
      double s = g2oCorrectedSiw.scale();

      eigt *= (1. / s);  //[R t/s;0 1]

      cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);  // 修正后

      // 注意，这里直接改变了 KF 位姿
      pKFi->SetPose(correctedTiw);

      // Make sure connections are updated
      pKFi->UpdateConnections();
    }

    /* 添加当前 KF 对匹配 KF MP 的观测 */
    // Start Loop Fusion
    // Update matched map points and replace if duplicated
    for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++) {
      if (mvpCurrentMatchedPoints[i]) {
        MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
        MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
        // 无则添加，有则替换
        if (pCurMP) pCurMP->Replace(pLoopMP);
        // 没有更新 pCurMP 的观测
        else {
          mpCurrentKF->AddMapPoint(pLoopMP, i);
          pLoopMP->AddObservation(mpCurrentKF, i);
          pLoopMP->ComputeDistinctiveDescriptors();
        }
      }
    }
  }

  /* 搜索并融合 */
  // Project MapPoints observed in the neighborhood of the loop keyframe
  // into the current keyframe and neighbors using corrected poses.
  // Fuse duplications.
  SearchAndFuse(CorrectedSim3);

  // After the MapPoint fusion, new links in the covisibility graph will appear
  // attaching both sides of the loop
  map<KeyFrame*, set<KeyFrame*> >
      LoopConnections;  // 回环连接  map<当前 KF 邻域 KF, set<匹配 KF 邻域 KF>>

  /* 遍历局部 KF，计算回环连接 */
  for (vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(),
                                   vend = mvpCurrentConnectedKFs.end();
       vit != vend; vit++) {
    KeyFrame* pKFi = *vit;
    vector<KeyFrame*> vpPreviousNeighbors =
        pKFi->GetVectorCovisibleKeyFrames();  // 修正前共视 KF

    // Update connections. Detect new links.

    // 更新共视关系，这里会依据对回环局部 MP 的观测，添加与匹配 KF
    // 邻域的共视关系
    pKFi->UpdateConnections();
    LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();

    // 避免将当前 KF 局部范围内的 KF 添加到回环连接 中
    for (vector<KeyFrame*>::iterator vit_prev = vpPreviousNeighbors.begin(),
                                     vend_prev = vpPreviousNeighbors.end();
         vit_prev != vend_prev; vit_prev++) {
      LoopConnections[pKFi].erase(*vit_prev);
    }
    for (vector<KeyFrame*>::iterator vit2 = mvpCurrentConnectedKFs.begin(),
                                     vend2 = mvpCurrentConnectedKFs.end();
         vit2 != vend2; vit2++) {
      LoopConnections[pKFi].erase(*vit2);
    }
  }

  /* 本质图优化 */
  // 本次优化中，当前 KF 和匹配 KF 的回环边是单独给出的
  // Optimize graph
  // Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF,
  //                                   NonCorrectedSim3, CorrectedSim3,
  //                                   LoopConnections, mbFixScale);
  if (mbFixScale) {
    CeresOptimizer::OptimizeEssentialGraphSim3(mpMap, mpMatchedKF, mpCurrentKF,
                                               NonCorrectedSim3, CorrectedSim3,
                                               LoopConnections);
  } else {
    CeresOptimizer::OptimizeEssentialGraphSE3(mpMap, mpMatchedKF, mpCurrentKF,
                                              NonCorrectedSim3, CorrectedSim3,
                                              LoopConnections);
  }

  mpMap->InformNewBigChange();

  /* 添加本质图回环边 */
  // Add loop edge
  mpMatchedKF->AddLoopEdge(mpCurrentKF);
  mpCurrentKF->AddLoopEdge(mpMatchedKF);

  /* 开启全局 BA 线程 */
  // Launch a new thread to perform Global Bundle Adjustment
  mbRunningGBA = true;
  mbFinishedGBA = false;
  mbStopGBA = false;

  mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this,
                           mpCurrentKF->mnId);

  // Loop closed. Release Local Mapping.
  mpLocalMapper->Release();

  mLastLoopKFid = mpCurrentKF->mnId;
}

/// @brief 搜索并融合回环 KF 地图点
/// @param CorrectedPosesMap 修正后局部 KF 位姿映射
void LoopClosing::SearchAndFuse(const KeyFrameAndPose& CorrectedPosesMap) {
  ORBmatcher matcher(0.8);

  // 遍历局部 KF 位姿映射
  for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(),
                                       mend = CorrectedPosesMap.end();
       mit != mend; mit++) {
    KeyFrame* pKF = mit->first;

    Sim3 g2oScw = mit->second;                   // 修正后位姿
    cv::Mat cvScw = Converter::toCvMat(g2oScw);  // 修正后位姿

    vector<MapPoint*> vpReplacePoints(
        mvpLoopMapPoints.size(),
        static_cast<MapPoint*>(NULL));  // 需要替换的 MP
    // 依据修正后的位姿匹配 KF 与回环局部 MP，获取需要替换的 MP
    matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

    // Get Map Mutex
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    const int nLP = mvpLoopMapPoints.size();
    // 替换 MP
    for (int i = 0; i < nLP; i++) {
      MapPoint* pRep = vpReplacePoints[i];
      if (pRep) {
        // 旧点替换新点，即匹配 KF 附近的 MP 替换当前 KF 附近的 MP
        pRep->Replace(mvpLoopMapPoints[i]);
      }
    }
  }
}

/// @brief 运行全局 BA
/// @param nLoopKF 回环 KF id  调用时输入当前 KF id
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF) {
  cout << "Starting Global Bundle Adjustment" << endl;

  int idx = mnFullBAIdx;

  // 运行全局 BA
  // Optimizer::GlobalBundleAdjustemnt(mpMap, 10, &mbStopGBA, nLoopKF, false);
  CeresOptimizer::GlobalBundleAdjustment(mpMap, 5, nLoopKF, false);

  /* 更新全部的 MP 和 KF */
  // 全局 BA 运行过程中，LocalMapping 中可能加入了新的 KF 和
  // MP，根据生成树修正他们的位姿 Update all MapPoints and KeyFrames Local
  // Mapping was active during BA, that means that there might be new keyframes
  // not included in the Global BA and they are not consistent with the updated
  // map. We need to propagate the correction through the spanning tree
  {
    unique_lock<mutex> lock(mMutexGBA);

    // 当前 GBA 是否是最新的 GBA
    if (idx != mnFullBAIdx) return;

    // GBA 正常结束，而不是被中断
    if (!mbStopGBA) {
      cout << "Global Bundle Adjustment finished" << endl;
      cout << "Updating map ..." << endl;

      // 暂停 LocalMapping
      mpLocalMapper->RequestStop();
      // Wait until Local Mapping has effectively stopped

      while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished()) {
        usleep(1000);
      }

      // Get Map Mutex
      unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

      /* 修正 KF 位姿 */
      // Correct keyframes starting at map first keyframe
      list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),
                                  mpMap->mvpKeyFrameOrigins.end());

      // 生成树广度优先遍历
      while (!lpKFtoCheck.empty()) {
        KeyFrame* pKF = lpKFtoCheck.front();              // 队首节点
        const set<KeyFrame*> sChilds = pKF->GetChilds();  // 子节点
        cv::Mat Twc = pKF->GetPoseInverse();  // 队首节点位姿 Twc
        // 恢复子节点位姿
        for (set<KeyFrame*>::const_iterator sit = sChilds.begin();
             sit != sChilds.end(); sit++) {
          KeyFrame* pChild = *sit;
          // 如果下面的条件成立，说明这个子 KF 是在 GBA 优化过程中新添加的
          if (pChild->mnBAGlobalForKF != nLoopKF) {
            // 修正位姿
            cv::Mat Tchildc = pChild->GetPose() * Twc;
            pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;  //*Tcorc*pKF->mTcwGBA;
            pChild->mnBAGlobalForKF = nLoopKF;
          }
          // 添加到队尾
          lpKFtoCheck.push_back(pChild);
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);
        lpKFtoCheck.pop_front();
      }

      /* 修正 MP 位置 */
      // Correct MapPoints
      const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

      // 遍历全部 MP
      for (size_t i = 0; i < vpMPs.size(); i++) {
        MapPoint* pMP = vpMPs[i];

        if (pMP->isBad()) continue;

        if (pMP->mnBAGlobalForKF == nLoopKF) {
          // If optimized by Global BA, just update
          pMP->SetWorldPos(pMP->mPosGBA);
        } else {
          // Update according to the correction of its reference keyframe
          KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

          // 应该不会出现，如果出现，代表参考 KF 位于生成树外
          if (pRefKF->mnBAGlobalForKF != nLoopKF) continue;

          // Map to non-corrected camera
          cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
          cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
          cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

          // Backproject using corrected camera
          cv::Mat Twc = pRefKF->GetPoseInverse();
          cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
          cv::Mat twc = Twc.rowRange(0, 3).col(3);

          pMP->SetWorldPos(Rwc * Xc + twc);
        }
      }

      mpMap->InformNewBigChange();

      mpLocalMapper->Release();

      cout << "Map updated!" << endl;
    }

    mbFinishedGBA = true;
    mbRunningGBA = false;
  }
}

////////////////////////////////////////////////////////////////////////////
/// 标志位设置与检查

/// @brief 请求重置
void LoopClosing::RequestReset() {
  {
    unique_lock<mutex> lock(mMutexReset);
    mbResetRequested = true;
  }

  while (1) {
    {
      unique_lock<mutex> lock2(mMutexReset);
      if (!mbResetRequested) break;
    }
    usleep(5000);
  }
}

/// @brief 重置检查
void LoopClosing::ResetIfRequested() {
  unique_lock<mutex> lock(mMutexReset);
  if (mbResetRequested) {
    mlpLoopKeyFrameQueue.clear();
    mLastLoopKFid = 0;
    mbResetRequested = false;
  }
}

/// @brief 请求终止
void LoopClosing::RequestFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

/// @brief 终止检查
/// @return
bool LoopClosing::CheckFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

/// @brief 设置终止
void LoopClosing::SetFinish() {
  unique_lock<mutex> lock(mMutexFinish);
  mbFinished = true;
}

/// @brief 是否已终止
/// @return
bool LoopClosing::isFinished() {
  unique_lock<mutex> lock(mMutexFinish);
  return mbFinished;
}

}  // namespace ORB_SLAM2
