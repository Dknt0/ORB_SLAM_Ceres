/**
 * CeresOptimizer
 *
 * Dknt 2024.3
 */

#ifndef CERESOPTIMIZER_H
#define CERESOPTIMIZER_H

#include <ceres/ceres.h>

#include "Frame.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"
#include "SlamManifold.h"
#include "SlamResidual.h"
#include "sim3.h"

#include <omp.h>

namespace ORB_SLAM2 {

class LoopClosing;
// class Tracking;
// class LocalMapping;
// class KeyFrameDatabase;

class CeresOptimizer {
 public:
  int static PoseOptimization(Frame *pFrame);

  void static LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap);

  int static OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2,
                          std::vector<MapPoint *> &vpMatches1, Sim3 &g2oS12,
                          const float th2, const bool bFixScale);

  void static OptimizeEssentialGraphSim3(
      Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
      const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
      const LoopClosing::KeyFrameAndPose &CorrectedSim3,
      const map<KeyFrame *, set<KeyFrame *> > &LoopConnections);

  void static OptimizeEssentialGraphSE3(
      Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
      const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
      const LoopClosing::KeyFrameAndPose &CorrectedSim3,
      const map<KeyFrame *, set<KeyFrame *> > &LoopConnections);

  void static GlobalBundleAdjustment(Map *pMap, int nIterations = 5,
                                     const unsigned long nLoopKF = 0,
                                     const bool bRobust = true);
};

}  // namespace ORB_SLAM2

#endif
