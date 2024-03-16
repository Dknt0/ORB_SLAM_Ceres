#include "CeresOptimizer.h"
/**
 * CeresOptimizer
 *
 * Dknt 2024.3
 */

namespace ORB_SLAM2 {

/// @brief Motion-Only Bundle-Adjustment used in tracking for fast estimation of
/// current pose. This function has been well tested.
///
/// @param pFrame Pointer to current frame
/// @return Number of inliers
int CeresOptimizer::PoseOptimization(Frame* pFrame) {
  /// 1. Setup Ceres optizimation problem
  ceres::Problem::Options problem_options;
  problem_options.enable_fast_removal = true;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem ceres_problem(problem_options);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 20;
  // TODO: More options
  ceres::Solver::Summary summary;

  pFrame->UpdateEigenPose();
  pFrame->tcw_opt_ = pFrame->tcw_;
  pFrame->qcw_opt_ = pFrame->qcw_;

  int init_correspondences = 0;
  size_t num_kp = pFrame->N;
  // To discard outlier measurements
  std::vector<std::shared_ptr<CostFunctionSE3ProjectionMonoPoseOnly>>
      v_mono_costfuncs;
  std::vector<ceres::ResidualBlockId> v_mono_residuals;
  std::vector<size_t> v_mono_pt_idx;
  std::vector<std::shared_ptr<CostFunctionSE3ProjectionStereoPoseOnly>>
      v_stereo_costfuncs;
  std::vector<ceres::ResidualBlockId> v_stereo_residuals;
  std::vector<size_t> v_stereo_pt_idx;

  v_mono_costfuncs.reserve(num_kp);
  v_mono_residuals.reserve(num_kp);
  v_mono_pt_idx.reserve(num_kp);
  v_stereo_costfuncs.reserve(num_kp);
  v_stereo_residuals.reserve(num_kp);
  v_stereo_pt_idx.reserve(num_kp);

  // LossFunction thresholds
  const double delta_mono = sqrt(5.991);
  const double delta_stereo = sqrt(7.815);
  ceres::LossFunction* lossfunction_mono = nullptr;
  ceres::LossFunction* lossfunction_stereo = nullptr;
  lossfunction_mono = new ceres::HuberLoss(delta_mono);
  lossfunction_stereo = new ceres::HuberLoss(delta_stereo);
  QuaternionManifoldIdentity* quaternion_manifold =
      new QuaternionManifoldIdentity();

  {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    for (size_t i = 0; i < num_kp; ++i) {
      MapPoint* mp = pFrame->mvpMapPoints[i];
      if (mp) {
        mp->UpdateEigenPose();
        if (pFrame->mvuRight[i] < 0) {
          /// Mono observation
          ++init_correspondences;
          pFrame->mvbOutlier[i] = false;
          // Create cost function object
          auto cost_ptr = new CostFunctionSE3ProjectionMonoPoseOnly(
              mp->twp_, pFrame->fx, pFrame->fy, pFrame->cx, pFrame->cy,
              Eigen::Vector2d(double(pFrame->mvKeysUn[i].pt.x),
                              double(pFrame->mvKeysUn[i].pt.y)));
          std::shared_ptr<CostFunctionSE3ProjectionMonoPoseOnly> cost(cost_ptr);
          // Information matrix
          const float inv_sigma2 =
              pFrame->mvInvLevelSigma2[pFrame->mvKeysUn[i].octave];
          cost->SetInformationMatrix(Eigen::Matrix2d::Identity() *
                                     sqrt(inv_sigma2));
          // Add residual to problem
          auto res_id = ceres_problem.AddResidualBlock(
              cost.get(), lossfunction_mono, pFrame->tcw_opt_.data(),
              pFrame->qcw_opt_.coeffs().data());
          // Save pointer to residual block and corresponding keypoint
          v_mono_costfuncs.push_back(cost);
          v_mono_residuals.push_back(res_id);
          v_mono_pt_idx.push_back(i);
          // Set manifold in the end
        } else {
          // Stereo observation
          ++init_correspondences;
          pFrame->mvbOutlier[i] = false;
          // Create cost function object
          auto cost_ptr = new CostFunctionSE3ProjectionStereoPoseOnly(
              mp->twp_, pFrame->fx, pFrame->fy, pFrame->cx, pFrame->cy,
              pFrame->mbf,
              Eigen::Vector3d(pFrame->mvKeysUn[i].pt.x,
                              pFrame->mvKeysUn[i].pt.y, pFrame->mvuRight[i]));
          std::shared_ptr<CostFunctionSE3ProjectionStereoPoseOnly> cost(
              cost_ptr);
          // Information matrix
          const float inv_sigma2 =
              pFrame->mvInvLevelSigma2[pFrame->mvKeysUn[i].octave];
          cost->SetInformationMatrix(Eigen::Matrix3d::Identity() *
                                     sqrt(inv_sigma2));
          // Add residual to problem
          auto res_id = ceres_problem.AddResidualBlock(
              cost.get(), lossfunction_stereo, pFrame->tcw_opt_.data(),
              pFrame->qcw_opt_.coeffs().data());
          // Save pointer to residual block and corresponding keypoint
          v_stereo_costfuncs.push_back(cost);
          v_stereo_residuals.push_back(res_id);
          v_stereo_pt_idx.push_back(i);
          // Set manifold in the end
        }
      }
    }
    ceres_problem.SetManifold(pFrame->qcw_opt_.coeffs().data(),
                              quaternion_manifold);
  }

  if (init_correspondences < 3) return 0;

  // Ceres::EvaluateResidualBlock returns half of error term without
  // lossfunction So we should set threshold below to half, this is different of
  // g2o
  const float chi2_mono = (5.991 / 2);
  const float chi2_stereo = (7.815 / 2);
  int num_bad_points = 0;

  /// Outliar discard
  ceres::Solve(options, &ceres_problem, &summary);
  // Mono residual discard
  for (size_t i = 0, i_max = v_mono_residuals.size(); i < i_max; ++i) {
    size_t pt_idx = v_mono_pt_idx[i];

    if (v_mono_residuals[i] != nullptr) {
      double res_error;
      ceres_problem.EvaluateResidualBlock(v_mono_residuals[i], false,
                                          &res_error, nullptr, nullptr);

      // if (i == 0) {
      //   std::cout << "Ceres first mono error " << res_error * 2 << std::endl;
      // }

      if (res_error > chi2_mono) {
        num_bad_points++;
        pFrame->mvbOutlier[pt_idx] = true;

        ceres_problem.RemoveResidualBlock(v_mono_residuals[i]);

        v_mono_residuals[i] = nullptr;
      } else {
        pFrame->mvbOutlier[pt_idx] = false;
        // TODO
      }
    } else {
      // TODO
    }
  }
  // Stereo residual discard
  for (size_t i = 0, i_max = v_stereo_residuals.size(); i < i_max; ++i) {
    size_t pt_idx = v_stereo_pt_idx[i];

    if (v_stereo_residuals[i] != nullptr) {
      double res_error;
      ceres_problem.EvaluateResidualBlock(v_stereo_residuals[i], false,
                                          &res_error, nullptr, nullptr);
      // if (i == 0) {
      //   std::cout << "Ceres first stereo error " << res_error * 2 <<
      //   std::endl;
      // }

      if (res_error > chi2_stereo) {
        num_bad_points++;
        pFrame->mvbOutlier[pt_idx] = true;

        ceres_problem.RemoveResidualBlock(v_stereo_residuals[i]);

        v_stereo_residuals[i] = nullptr;
      } else {
        pFrame->mvbOutlier[pt_idx] = false;
        // TODO
      }
    } else {
      // TODO
    }
  }
  if (ceres_problem.NumResidualBlocks() > 10) {
    pFrame->tcw_opt_ = pFrame->tcw_;
    pFrame->qcw_opt_ = pFrame->qcw_;
    ceres::Solve(options, &ceres_problem, &summary);
  }

  pFrame->tcw_ = pFrame->tcw_opt_;
  pFrame->qcw_ = pFrame->qcw_opt_;

  pFrame->UpdatePoseFromEigen();
  pFrame->SetPose(pFrame->mTcw);

  // break here
  return init_correspondences - num_bad_points;
}

// This function need to be refined
void CeresOptimizer::LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag,
                                           Map* pMap) {
  // Set ceres problem
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem ceres_problem(problem_options);
  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 5;
  ceres::Solver::Summary summary;

  // Declear variables
  std::list<KeyFrame*> list_local_kfs;
  std::list<MapPoint*> list_local_mps;
  std::vector<ceres::ResidualBlockId> vec_res_mono;
  std::vector<ceres::ResidualBlockId> vec_res_stereo;
  std::vector<std::shared_ptr<CostFunctionSE3ProjectionMono>> vec_cost_mono;
  std::vector<std::shared_ptr<CostFunctionSE3ProjectionStereo>> vec_cost_stereo;
  QuaternionManifoldIdentity* manifold = new QuaternionManifoldIdentity;

  // Get local KFs
  pKF->mnBALocalForKF_temp = pKF->mnId;
  list_local_kfs.push_back(pKF);
  cv::Mat Tcw_pkf = pKF->GetPose();
  Eigen::Isometry3d Tcw_eigen_pkf;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      Tcw_eigen_pkf(i, j) = Tcw_pkf.at<float>(i, j);
    }
  }
  pKF->tcw_lba_opt = Tcw_eigen_pkf.translation();
  pKF->qcw_lba_opt = Eigen::Quaterniond(Tcw_eigen_pkf.rotation());

  std::vector<KeyFrame*> covisibility_kf = pKF->GetVectorCovisibleKeyFrames();
  // Neet to pre-allocate memories for vector
  for (size_t i = 0, imax = covisibility_kf.size(); i < imax; ++i) {
    KeyFrame* kf = covisibility_kf[i];
    kf->mnBALocalForKF_temp = pKF->mnId;
    if (kf->isBad()) continue;
    list_local_kfs.push_back(kf);
    // Add local optimization variables
    cv::Mat Tcw = kf->GetPose();
    Eigen::Isometry3d Tcw_eigen;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        Tcw_eigen(i, j) = Tcw.at<float>(i, j);
      }
    }
    kf->tcw_lba_opt = Tcw_eigen.translation();
    kf->qcw_lba_opt = Eigen::Quaterniond(Tcw_eigen.rotation());
    // Add parameter block to Ceres problem
    ceres_problem.AddParameterBlock(kf->tcw_lba_opt.data(), 3);
    ceres_problem.AddParameterBlock(kf->qcw_lba_opt.coeffs().data(), 4,
                                    manifold);
  }

  // Get local MPs
  for (std::list<KeyFrame*>::iterator it_list = list_local_kfs.begin(),
                                      it_list_end = list_local_kfs.end();
       it_list != it_list_end; ++it_list) {
    std::vector<MapPoint*> mps = (*it_list)->GetMapPointMatches();

    for (size_t i = 0, imax = mps.size(); i < imax; ++i) {
      MapPoint* mp = mps[i];
      if (!mp || mp->isBad() || mp->mnBALocalForKF_temp == pKF->mnId) continue;

      list_local_mps.push_back(mp);
      mp->mnBALocalForKF_temp = pKF->mnId;
      cv::Mat twp_cv = mp->GetWorldPos();
      mp->UpdateEigenPose();
      Eigen::Vector3d twp;
      for (size_t i = 0; i < 3; ++i) {
        twp[i] = twp_cv.at<float>(i, 0);
      }
      mp->twp_lba_opt = twp;
      // Add parameter block to Ceres problem
      ceres_problem.AddParameterBlock(mp->twp_lba_opt.data(), 3);
    }
  }

  // Get fixed local KFs
  for (std::list<MapPoint*>::iterator it_list = list_local_mps.begin(),
                                      it_list_end = list_local_mps.end();
       it_list != it_list_end; ++it_list) {
    MapPoint* mp = *it_list;
    std::map<KeyFrame*, size_t> kfs = mp->GetObservations();
    for (std::map<KeyFrame*, size_t>::iterator it_map = kfs.begin(),
                                               it_map_end = kfs.end();
         it_map != it_map_end; ++it_map) {
      KeyFrame* kf = it_map->first;
      if (kf->isBad() || kf->mnBALocalForKF_temp == pKF->mnId ||
          kf->mnBAFixedForKF_temp == pKF->mnId)
        continue;
      kf->mnBAFixedForKF_temp = pKF->mnId;
      // Add local optimization variables
      cv::Mat Tcw = kf->GetPose();
      Eigen::Isometry3d Tcw_eigen;
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          Tcw_eigen(i, j) = Tcw.at<float>(i, j);
        }
      }
      kf->tcw_lba_opt = Tcw_eigen.translation();
      kf->qcw_lba_opt = Eigen::Quaterniond(Tcw_eigen.rotation());
      // Add parameter block to Ceres problem
      ceres_problem.AddParameterBlock(kf->tcw_lba_opt.data(), 3);
      ceres_problem.AddParameterBlock(kf->qcw_lba_opt.coeffs().data(), 4);
      // Set parameter constant
      ceres_problem.SetParameterBlockConstant(kf->tcw_lba_opt.data());
      ceres_problem.SetParameterBlockConstant(kf->qcw_lba_opt.coeffs().data());
    }
  }

  // Create loss function
  const double delta_mono = sqrt(5.991);
  const double delta_stereo = sqrt(7.815);
  ceres::LossFunction* lossfunction_mono = nullptr;
  ceres::LossFunction* lossfunction_stereo = nullptr;
  lossfunction_mono = new ceres::HuberLoss(delta_mono);
  lossfunction_stereo = new ceres::HuberLoss(delta_stereo);

  // Add residual blocks
  for (std::list<MapPoint*>::iterator it_list = list_local_mps.begin(),
                                      it_list_end = list_local_mps.end();
       it_list != it_list_end; ++it_list) {
    MapPoint* mp = *it_list;
    std::map<KeyFrame*, size_t> kfs = mp->GetObservations();
    for (std::map<KeyFrame*, size_t>::iterator it_map = kfs.begin(),
                                               it_map_end = kfs.end();
         it_map != it_map_end; ++it_map) {
      KeyFrame* kf = it_map->first;
      if (kf->isBad()) continue;
      size_t kp_id = it_map->second;
      cv::KeyPoint kp = kf->mvKeysUn[kp_id];

      if (kf->mvuRight[kp_id] < 0) {
        // Mono observation
        // Create cost function object
        CostFunctionSE3ProjectionMono* cost_ptr =
            new CostFunctionSE3ProjectionMono(
                kf->fx, kf->fy, kf->cx, kf->cy,
                Eigen::Vector2d(kp.pt.x, kp.pt.y));
        cost_ptr->SetKeyFrameAndMapPoint(kf, mp);
        // Make shared pointer
        std::shared_ptr<CostFunctionSE3ProjectionMono> cost(cost_ptr);
        // Information matrix
        const float inv_sigma2 = kf->mvInvLevelSigma2[kp.octave];
        cost_ptr->SetInformationMatrix(Eigen::Matrix2d::Identity() *
                                       sqrt(inv_sigma2));
        // Add residual to problem
        ceres::ResidualBlockId res_id = ceres_problem.AddResidualBlock(
            cost_ptr, lossfunction_mono, kf->tcw_lba_opt.data(),
            kf->qcw_lba_opt.coeffs().data(), mp->twp_lba_opt.data());
        // Save to vector
        vec_cost_mono.push_back(cost);
        vec_res_mono.push_back(res_id);
        // Manifold has already been set
      } else {
        // Stereo obserbation
        // Create cost function object
        CostFunctionSE3ProjectionStereo* cost_ptr =
            new CostFunctionSE3ProjectionStereo(
                kf->fx, kf->fy, kf->cx, kf->cy, kf->mbf,
                Eigen::Vector3d(kp.pt.x, kp.pt.y, kf->mvuRight[kp_id]));
        cost_ptr->SetKeyFrameAndMapPoint(kf, mp);
        // Make shared pointer
        std::shared_ptr<CostFunctionSE3ProjectionStereo> cost(cost_ptr);
        // Information matrix
        const float inv_sigma2 = kf->mvInvLevelSigma2[kp.octave];
        cost_ptr->SetInformationMatrix(Eigen::Matrix3d::Identity() *
                                       sqrt(inv_sigma2));
        // Add residual to problem
        ceres::ResidualBlockId res_id = ceres_problem.AddResidualBlock(
            cost_ptr, lossfunction_stereo, kf->tcw_lba_opt.data(),
            kf->qcw_lba_opt.coeffs().data(), mp->twp_lba_opt.data());
        // Save to vector
        vec_cost_stereo.push_back(cost);
        vec_res_stereo.push_back(res_id);
        // Manifold has already been set
      }
    }
  }

  // if (pbStopFlag && *pbStopFlag) {
  //   return;
  // }

  // First optimization
  ceres::Solve(options, &ceres_problem, &summary);
  // auto res_num_1 = ceres_problem.NumResidualBlocks();

  // Discard outliars
  const float chi2_mono = (5.991 / 2);
  const float chi2_stereo = (7.815 / 2);
  std::vector<std::pair<KeyFrame*, MapPoint*>> vec_to_erase;
  for (size_t i = 0, imax = vec_res_mono.size(); i < imax; ++i) {
    // Depth shoud be positive
    std::vector<double*> parameter_block;
    ceres_problem.GetParameterBlocksForResidualBlock(vec_res_mono[i],
                                                     &parameter_block);
    Eigen::Map<const Eigen::Vector3d> tcw(parameter_block[0]);
    Eigen::Map<const Eigen::Quaterniond> qcw(parameter_block[1]);
    Eigen::Map<const Eigen::Vector3d> twq(parameter_block[2]);
    Eigen::Vector3d tcq = qcw * twq + tcw;
    // Error should be less than threshold
    double error;
    ceres_problem.EvaluateResidualBlock(vec_res_mono[i], false, &error, nullptr,
                                        nullptr);

    if (error > chi2_mono || tcq[2] < 0) {
      ceres_problem.RemoveResidualBlock(vec_res_mono[i]);
      auto cost = vec_cost_mono[i];
      vec_to_erase.push_back(
          std::make_pair(static_cast<KeyFrame*>(cost->kf_ptr_),
                         static_cast<MapPoint*>(cost->mp_ptr_)));
    }
  }
  for (size_t i = 0, imax = vec_res_stereo.size(); i < imax; ++i) {
    // Depth shoud be positive
    std::vector<double*> parameter_block;
    ceres_problem.GetParameterBlocksForResidualBlock(vec_res_stereo[i],
                                                     &parameter_block);
    Eigen::Map<const Eigen::Vector3d> tcw(parameter_block[0]);
    Eigen::Map<const Eigen::Quaterniond> qcw(parameter_block[1]);
    Eigen::Map<const Eigen::Vector3d> twq(parameter_block[2]);
    Eigen::Vector3d tcq = qcw * twq + tcw;
    // Error should be less than threshold
    double error;
    ceres_problem.EvaluateResidualBlock(vec_res_stereo[i], false, &error,
                                        nullptr, nullptr);
    if (error > chi2_stereo || tcq[2] < 0) {
      ceres_problem.RemoveResidualBlock(vec_res_stereo[i]);
      auto cost = vec_cost_stereo[i];
      vec_to_erase.push_back(
          std::make_pair(static_cast<KeyFrame*>(cost->kf_ptr_),
                         static_cast<MapPoint*>(cost->mp_ptr_)));
    }
  }

  // Second optimization
  if (!pbStopFlag || !(*pbStopFlag)) {
    options.max_num_iterations = 5;
    ceres::Solve(options, &ceres_problem, &summary);
    // auto res_num_2 = ceres_problem.NumResidualBlocks();
  } else {
    // std::cout << "[LBA] with out second optimization!" << std::endl; 
  }

  // Delete observations
  std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);
  if (!vec_to_erase.empty()) {
    for (size_t i = 0, imax = vec_to_erase.size(); i < imax; ++i) {
      KeyFrame* kf = vec_to_erase[i].first;
      MapPoint* mp = vec_to_erase[i].second;
      kf->EraseMapPointMatch(mp);
      mp->EraseObservation(kf);
    }
  }

  // Recover KF data
  for (std::list<KeyFrame*>::iterator it_list = list_local_kfs.begin(),
                                      it_list_end = list_local_kfs.end();
       it_list != it_list_end; ++it_list) {
    KeyFrame* kf = *it_list;
    cv::Mat Tcw(4, 4, CV_32F);
    Eigen::Isometry3d Tcw_eigen(kf->qcw_lba_opt);
    Tcw_eigen.pretranslate(kf->tcw_lba_opt);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        Tcw.at<float>(i, j) = Tcw_eigen(i, j);
      }
    }
    kf->SetPose(Tcw);
  }

  // Recover MP data
  for (std::list<MapPoint*>::iterator it_list = list_local_mps.begin(),
                                      it_list_end = list_local_mps.end();
       it_list != it_list_end; ++it_list) {
    MapPoint* mp = *it_list;
    cv::Mat pw_cv(3, 1, CV_32F);
    for (size_t i = 0; i < 3; ++i) {
      pw_cv.at<float>(i, 0) = mp->twp_lba_opt[i];
    }
    mp->SetWorldPos(pw_cv);
  }
}

int CeresOptimizer::OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
                 std::vector<MapPoint*>& vpMatches1, Sim3& g2oS12,
                 const float th2, const bool bFixScale) {
  // Set Ceres problem
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::TAKE_OWNERSHIP;
  ceres::Problem ceres_problem(problem_options);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 5;
  ceres::Solver::Summary summary;

  // Prepare variables
  Sim3 S12_opt = g2oS12;
  size_t num_pt = vpMatches1.size();
  std::vector<ceres::ResidualBlockId> vec_forward_res;
  std::vector<ceres::ResidualBlockId> vec_inverse_res;
  std::vector<size_t> vec_kf_idx;  // To discard outlizers in vpMatches1
  vec_forward_res.reserve(num_pt);
  vec_inverse_res.reserve(num_pt);

  std::vector<MapPoint*> vec_pts_kf1 = pKF1->GetMapPointMatches();
  cv::Mat R1w = pKF1->GetRotation();
  cv::Mat t1w = pKF1->GetTranslation();
  cv::Mat R2w = pKF2->GetRotation();
  cv::Mat t2w = pKF2->GetTranslation();

  ceres::LossFunction* lossfunction = nullptr;
  lossfunction = new ceres::HuberLoss(sqrt(th2));
  QuaternionManifoldIdentity* manifold = new QuaternionManifoldIdentity;
  for (size_t i = 0, imax = vpMatches1.size(); i < imax; ++i) {
    MapPoint* mp1 = vec_pts_kf1[i];
    MapPoint* mp2 = vpMatches1[i];
    if (!mp1 || !mp2) continue;
    size_t mp2_idx = mp2->GetIndexInKeyFrame(pKF2);
    if (mp1->isBad() || mp2->isBad() || mp2_idx < 0) continue;

    // Forward observation
    {
      cv::Mat p2w = mp2->GetWorldPos();
      cv::Mat p2c = R2w * p2w + t2w;
      Eigen::Vector3d p2c_eigen(p2c.at<float>(0), p2c.at<float>(1),
                                p2c.at<float>(2));
      cv::KeyPoint kp1 = pKF1->mvKeysUn[i];
      CostFunctionSim3ProjectionMono* cost_forward =
          new CostFunctionSim3ProjectionMono(
              p2c_eigen, pKF1->fx, pKF1->fy, pKF1->cx, pKF1->cy,
              Eigen::Vector2d(kp1.pt.x, kp1.pt.y));
      const float inv_sigma2_kf1 =
          pKF1->mvInvLevelSigma2[pKF1->mvKeysUn[i].octave];
      cost_forward->SetInformationMatrix(Eigen::Matrix2d::Identity() *
                                         sqrt(inv_sigma2_kf1));
      ceres::ResidualBlockId forward_res_id = ceres_problem.AddResidualBlock(
          cost_forward, lossfunction, S12_opt.t.data(),
          S12_opt.r.coeffs().data(), &(S12_opt.s));
      vec_forward_res.push_back(forward_res_id);
    }

    // Inverse observation
    {
      cv::Mat p1w = mp1->GetWorldPos();
      cv::Mat p1c = R1w * p1w + t1w;
      Eigen::Vector3d p1c_eigen(p1c.at<float>(0), p1c.at<float>(1),
                                p1c.at<float>(2));
      cv::KeyPoint kp2 = pKF2->mvKeysUn[mp2_idx];

      CostFunctionSim3InverseProjectionMono* cost_inverse =
          new CostFunctionSim3InverseProjectionMono(
              p1c_eigen, pKF2->fx, pKF2->fy, pKF2->cx, pKF2->cy,
              Eigen::Vector2d(kp2.pt.x, kp2.pt.y));
      const float inv_sigma2_kf2 =
          pKF2->mvInvLevelSigma2[pKF2->mvKeysUn[mp2_idx].octave];
      cost_inverse->SetInformationMatrix(Eigen::Matrix2d::Identity() *
                                         sqrt(inv_sigma2_kf2));
      ceres::ResidualBlockId inverse_res_id = ceres_problem.AddResidualBlock(
          cost_inverse, lossfunction, S12_opt.t.data(),
          S12_opt.r.coeffs().data(), &(S12_opt.s));
      vec_inverse_res.push_back(inverse_res_id);
    }

    // Save keypoint id
    vec_kf_idx.push_back(i);
  }
  ceres_problem.SetManifold(S12_opt.r.coeffs().data(), manifold);

  // Optimization 1
  ceres::Solve(options, &ceres_problem, &summary);
  std::cout << "[OptimizeSim3] " << summary.BriefReport() << std::endl;

  // Discard outlizer 1
  int num_bad = 0;
  const double threshold = th2 / 2;
  for (size_t i = 0, imax = vec_forward_res.size(); i < imax; ++i) {
    double e_forward;
    double e_inverse;
    ceres_problem.EvaluateResidualBlock(vec_forward_res[i], false, &e_forward,
                                        nullptr, nullptr);
    ceres_problem.EvaluateResidualBlock(vec_inverse_res[i], false, &e_inverse,
                                        nullptr, nullptr);
    if (e_forward > threshold || e_inverse > threshold) {
      ceres_problem.RemoveResidualBlock(vec_forward_res[i]);
      ceres_problem.RemoveResidualBlock(vec_inverse_res[i]);
      vec_forward_res[i] = nullptr;
      vec_inverse_res[i] = nullptr;
      vpMatches1[vec_kf_idx[i]] = nullptr;
      num_bad++;
    }
  }
  if (vec_forward_res.size() - num_bad < 10) return 0;

  // Optimization 2
  ceres::Solve(options, &ceres_problem, &summary);

  // Discard outlizer 2
  int num_good = 0;
  for (size_t i = 0, imax = vec_forward_res.size(); i < imax; ++i) {
    if (!vec_forward_res[i]) continue;
    double e_forward;
    double e_inverse;
    ceres_problem.EvaluateResidualBlock(vec_forward_res[i], false, &e_forward,
                                        nullptr, nullptr);
    ceres_problem.EvaluateResidualBlock(vec_inverse_res[i], false, &e_inverse,
                                        nullptr, nullptr);
    if (e_forward > threshold || e_inverse > threshold) {
      ceres_problem.RemoveResidualBlock(vec_forward_res[i]);
      ceres_problem.RemoveResidualBlock(vec_inverse_res[i]);
      vec_forward_res[i] = nullptr;
      vec_inverse_res[i] = nullptr;
      vpMatches1[vec_kf_idx[i]] = nullptr;
    } else {
      num_good++;
    }
  }

  // Recover data
  g2oS12 = S12_opt;


  return num_good;
}

void CeresOptimizer::OptimizeEssentialGraphSim3(
    Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
    const LoopClosing::KeyFrameAndPose& NonCorrectedSim3,
    const LoopClosing::KeyFrameAndPose& CorrectedSim3,
    const map<KeyFrame*, set<KeyFrame*>>& LoopConnections) {
  // std::cout << "Start ceres Sim3 Essential Graph optimization" << std::endl;

  // Solver setting
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::TAKE_OWNERSHIP;
  ceres::Problem ceres_problem(problem_options);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 5;
  // options.num_threads = 1;
  ceres::Solver::Summary summary;

  // init lambda?

  const std::vector<KeyFrame*> vec_all_kf = pMap->GetAllKeyFrames();
  const std::vector<MapPoint*> vec_all_mp = pMap->GetAllMapPoints();
  // const unsigned long int max_kf_id = pMap->GetMaxKFid();
  const int min_feat = 100;

  // Prepare pose variable for each KF
  for (size_t i = 0, imax = vec_all_kf.size(); i < imax; ++i) {
    KeyFrame* kf = vec_all_kf[i];
    if (kf->isBad()) continue;
    LoopClosing::KeyFrameAndPose::const_iterator corrected_id =
        CorrectedSim3.find(kf);
    if (corrected_id != CorrectedSim3.end()) {
      // Use corrected pose Scw if this KF is in the covisiable group w.r.t.
      // current KF
      kf->Scw_ = corrected_id->second;
      kf->Scw_opt_ = kf->Scw_;
    } else {
      // Otherwise use original pose
      kf->UpdateEigenPose();
      kf->Scw_ = Sim3(kf->qcw_.matrix(), kf->tcw_, 1.0);
      kf->Scw_opt_ = kf->Scw_;
    }
  }

  std::set<std::pair<long unsigned int, long unsigned int>> inserted_edges;
  ceres::LossFunction* lossfunction = nullptr;
  lossfunction = new ceres::HuberLoss(1.0);
  QuaternionManifoldIdentity* so3_manifold = new QuaternionManifoldIdentity();

  // Add residual blocks
  Eigen::Matrix<double, 7, 7> info_matrix;
  info_matrix.setIdentity();
  // Add current loop edges from input map
  for (std::map<KeyFrame*, std::set<KeyFrame*>>::const_iterator
           mit = LoopConnections.begin(),
           mit_end = LoopConnections.end();
       mit != mit_end; ++mit) {
    KeyFrame* kf1 = mit->first;
    long unsigned int kf1_id = kf1->mnId;
    const std::set<KeyFrame*>& kf_connections = mit->second;
    Sim3 S1w = kf1->Scw_;

    for (std::set<KeyFrame*>::const_iterator sit = kf_connections.begin(),
                                             sit_end = kf_connections.end();
         sit != sit_end; ++sit) {
      KeyFrame* kf2 = *sit;
      long unsigned int kf2_id = kf2->mnId;
      // Add edges with enough covisibility and all edges connected with
      // pCurKF or pLoopKF
      if ((kf1_id != pCurKF->mnId || kf2_id != pLoopKF->mnId) &&
          kf1->GetWeight(kf2) < min_feat)
        continue;
      // Compute relative pose, in fact these are corrected ones
      Sim3 S2w = kf2->Scw_;
      Sim3 S12 = S1w * S2w.inverse();
      // Create cost function object
      CostFunctionPoseGraphSim3* costfunction =
          new CostFunctionPoseGraphSim3(S12.t, S12.r, S12.s);
      costfunction->SetInformationMatrix(info_matrix);
      // Add residual block
      ceres_problem.AddResidualBlock(
          costfunction, lossfunction, kf1->Scw_opt_.t.data(),
          kf1->Scw_opt_.r.coeffs().data(), &(kf1->Scw_opt_.s),
          kf2->Scw_opt_.t.data(), kf2->Scw_opt_.r.coeffs().data(),
          &(kf2->Scw_opt_.s));
      // Set SO3 manifold
      ceres_problem.SetManifold(kf1->Scw_opt_.r.coeffs().data(), so3_manifold);
      ceres_problem.SetManifold(kf2->Scw_opt_.r.coeffs().data(), so3_manifold);
      inserted_edges.insert(
          std::make_pair(min(kf1_id, kf2_id), max(kf1_id, kf2_id)));
    }
  }

  // Add essential graph edges from covisibility graph and spanning tree
  for (size_t i = 0, imax = vec_all_kf.size(); i < imax; ++i) {
    KeyFrame* kf1 = vec_all_kf[i];
    // long unsigned int kf1_id = kf1->mnId;
    Sim3 S1w;
    LoopClosing::KeyFrameAndPose::const_iterator pose_it =
        NonCorrectedSim3.find(kf1);
    if (pose_it != NonCorrectedSim3.end()) {
      S1w = pose_it->second;
    } else {
      S1w = kf1->Scw_;
    }

    // Spanning tree edge
    KeyFrame* parent_kf = kf1->GetParent();
    if (parent_kf) {
      Sim3 Spw;
      LoopClosing::KeyFrameAndPose::const_iterator pose_it_p =
          NonCorrectedSim3.find(parent_kf);
      if (pose_it_p != NonCorrectedSim3.end()) {
        Spw = pose_it_p->second;
      } else {
        Spw = parent_kf->Scw_;
      }
      Sim3 S1p = S1w * Spw.inverse();
      // Create cost function object
      CostFunctionPoseGraphSim3* costfunction =
          new CostFunctionPoseGraphSim3(S1p.t, S1p.r, S1p.s);
      costfunction->SetInformationMatrix(info_matrix);
      // Add residual block
      ceres_problem.AddResidualBlock(
          costfunction, lossfunction, kf1->Scw_opt_.t.data(),
          kf1->Scw_opt_.r.coeffs().data(), &(kf1->Scw_opt_.s),
          parent_kf->Scw_opt_.t.data(), parent_kf->Scw_opt_.r.coeffs().data(),
          &(parent_kf->Scw_opt_.s));
      // Set SO3 manifold
      ceres_problem.SetManifold(kf1->Scw_opt_.r.coeffs().data(), so3_manifold);
      ceres_problem.SetManifold(parent_kf->Scw_opt_.r.coeffs().data(),
                                so3_manifold);
    }

    // Loop edges
    const set<KeyFrame*> loop_kfs = kf1->GetLoopEdges();
    for (set<KeyFrame*>::const_iterator it_kf2 = loop_kfs.begin(),
                                        it_kf2_end = loop_kfs.end();
         it_kf2 != it_kf2_end; it_kf2++) {
      KeyFrame* kf2 = *it_kf2;
      if (kf2->mnId < kf1->mnId) {
        Sim3 S2w;
        LoopClosing::KeyFrameAndPose::const_iterator pose_it_2 =
            NonCorrectedSim3.find(kf2);
        if (pose_it_2 != NonCorrectedSim3.end()) {
          S2w = pose_it_2->second;
        } else {
          S2w = kf2->Scw_;
        }
        Sim3 S12 = S1w * S2w.inverse();
        // Create cost function object
        CostFunctionPoseGraphSim3* costfunction =
            new CostFunctionPoseGraphSim3(S12.t, S12.r, S12.s);
        costfunction->SetInformationMatrix(info_matrix);
        // Add residual block
        ceres_problem.AddResidualBlock(
            costfunction, lossfunction, kf1->Scw_opt_.t.data(),
            kf1->Scw_opt_.r.coeffs().data(), &(kf1->Scw_opt_.s),
            kf2->Scw_opt_.t.data(), kf2->Scw_opt_.r.coeffs().data(),
            &(kf2->Scw_opt_.s));
        // Set SO3 manifold
        ceres_problem.SetManifold(kf1->Scw_opt_.r.coeffs().data(),
                                  so3_manifold);
        ceres_problem.SetManifold(kf2->Scw_opt_.r.coeffs().data(),
                                  so3_manifold);
      }
    }

    // Covisibility graph edges
    vector<KeyFrame*> connected_kfs = kf1->GetCovisiblesByWeight(min_feat);
    for (size_t i = 0, imax = connected_kfs.size(); i < imax; ++i) {
      KeyFrame* kf2 = connected_kfs[i];
      if (kf2 && kf2 != parent_kf && !kf1->hasChild(kf2) &&
          !loop_kfs.count(kf2)) {
        if (!kf2->isBad() && kf2->mnId < kf1->mnId) {
          if (inserted_edges.count(
                  std::make_pair(std::min(kf1->mnId, kf2->mnId),
                                 std::max(kf1->mnId, kf2->mnId))))
            continue;
          Sim3 S2w;
          LoopClosing::KeyFrameAndPose::const_iterator pose_it_2 =
              NonCorrectedSim3.find(kf2);
          if (pose_it_2 != NonCorrectedSim3.end()) {
            S2w = pose_it_2->second;
          } else {
            S2w = kf2->Scw_;
          }
          Sim3 S12 = S1w * S2w.inverse();
          // Create cost function object
          CostFunctionPoseGraphSim3* costfunction =
              new CostFunctionPoseGraphSim3(S12.t, S12.r, S12.s);
          costfunction->SetInformationMatrix(info_matrix);
          // Add residual block
          ceres_problem.AddResidualBlock(
              costfunction, lossfunction, kf1->Scw_opt_.t.data(),
              kf1->Scw_opt_.r.coeffs().data(), &(kf1->Scw_opt_.s),
              kf2->Scw_opt_.t.data(), kf2->Scw_opt_.r.coeffs().data(),
              &(kf2->Scw_opt_.s));
          // Set SO3 manifold
          ceres_problem.SetManifold(kf1->Scw_opt_.r.coeffs().data(),
                                    so3_manifold);
          ceres_problem.SetManifold(kf2->Scw_opt_.r.coeffs().data(),
                                    so3_manifold);
        }
      }
    }
  }

  // std::cout << "NumResidualBlocks: " << ceres_problem.NumResidualBlocks()
  //           << std::endl;
  // Set LoopKF as fixed
  ceres_problem.SetParameterBlockConstant(pLoopKF->Scw_opt_.r.coeffs().data());
  ceres_problem.SetParameterBlockConstant(pLoopKF->Scw_opt_.t.data());
  ceres_problem.SetParameterBlockConstant(&(pLoopKF->Scw_opt_.s));

  ceres::Solve(options, &ceres_problem, &summary);

  std::cout << summary.BriefReport()
            << "Used time: " << summary.total_time_in_seconds << " sec "
            << std::endl;

  std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);
  /// Recovery KF data
  for (size_t i = 0, imax = vec_all_kf.size(); i < imax; ++i) {
    KeyFrame* kf = vec_all_kf[i];
    if (kf->isBad()) continue;
    Sim3& Scw = kf->Scw_opt_;
    Eigen::Matrix3d Rcw = Scw.r.matrix();
    double scw = Scw.s;
    Eigen::Vector3d tcw = Scw.t / scw;

    Eigen::Isometry3d Tcw_eigen(Rcw);
    Tcw_eigen.pretranslate(tcw);
    cv::Mat Tcw(4, 4, CV_32F);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        Tcw.at<float>(i, j) = Tcw_eigen(i, j);
      }
    }
    kf->SetPose(Tcw);
  }

  /// Recovery MP data
  for (size_t i = 0, imax = vec_all_mp.size(); i < imax; ++i) {
    MapPoint* mp = vec_all_mp[i];
    if (mp->isBad()) continue;
    Sim3 Srw_old;
    Sim3 Srw_new;
    if (mp->mnCorrectedByKF == pCurKF->mnId) {
      Srw_old = mp->mnCorrectedReferencePtr->Scw_;
      Srw_new = mp->mnCorrectedReferencePtr->Scw_opt_;
    } else {
      Srw_old = mp->GetReferenceKeyFrame()->Scw_;
      Srw_new = mp->GetReferenceKeyFrame()->Scw_opt_;
    }
    cv::Mat twp_cv = mp->GetWorldPos();
    Eigen::Vector3d twp_old(twp_cv.at<float>(0), twp_cv.at<float>(01),
                            twp_cv.at<float>(2));
    Eigen::Vector3d twp_new = Srw_new.inverse().map(Srw_old.map(twp_old));
    cv::Mat twp_new_cv(3, 1, CV_32F);
    twp_new_cv.at<float>(0) = twp_new(0);
    twp_new_cv.at<float>(1) = twp_new(1);
    twp_new_cv.at<float>(2) = twp_new(2);
    mp->SetWorldPos(twp_new_cv);
    mp->UpdateNormalAndDepth();
  }
}

void CeresOptimizer::OptimizeEssentialGraphSE3(
    Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
    const LoopClosing::KeyFrameAndPose& NonCorrectedSim3,
    const LoopClosing::KeyFrameAndPose& CorrectedSim3,
    const map<KeyFrame*, set<KeyFrame*>>& LoopConnections) {
  // std::cout << "Start ceres SE3 Essential Graph optimization" << std::endl;

  // Solver setting
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::TAKE_OWNERSHIP;
  ceres::Problem ceres_problem(problem_options);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 5;
  // options.num_threads = 1;
  ceres::Solver::Summary summary;

  // init lambda?

  const std::vector<KeyFrame*> vec_all_kf = pMap->GetAllKeyFrames();
  const std::vector<MapPoint*> vec_all_mp = pMap->GetAllMapPoints();
  // const unsigned long int max_kf_id = pMap->GetMaxKFid();
  const int min_feat = 100;

  // Prepare pose variable for each KF
  for (size_t i = 0, imax = vec_all_kf.size(); i < imax; ++i) {
    KeyFrame* kf = vec_all_kf[i];
    if (kf->isBad()) continue;
    LoopClosing::KeyFrameAndPose::const_iterator corrected_id =
        CorrectedSim3.find(kf);
    if (corrected_id != CorrectedSim3.end()) {
      // Use corrected pose Scw if this KF is in the covisiable group w.r.t.
      // current KF
      kf->Scw_ = corrected_id->second;
      kf->Scw_opt_ = kf->Scw_;
    } else {
      // Otherwise use original pose
      kf->UpdateEigenPose();
      kf->Scw_ = Sim3(kf->qcw_.matrix(), kf->tcw_, 1.0);
      kf->Scw_opt_ = kf->Scw_;
    }
  }

  std::set<std::pair<long unsigned int, long unsigned int>> inserted_edges;
  ceres::LossFunction* lossfunction = nullptr;
  lossfunction = new ceres::HuberLoss(1.0);
  QuaternionManifoldIdentity* so3_manifold = new QuaternionManifoldIdentity();

  // Add residual blocks
  Eigen::Matrix<double, 6, 6> info_matrix;
  info_matrix.setIdentity();
  // Add current loop edges from input map
  for (std::map<KeyFrame*, std::set<KeyFrame*>>::const_iterator
           mit = LoopConnections.begin(),
           mit_end = LoopConnections.end();
       mit != mit_end; ++mit) {
    KeyFrame* kf1 = mit->first;
    long unsigned int kf1_id = kf1->mnId;
    const std::set<KeyFrame*>& kf_connections = mit->second;
    Sim3 S1w = kf1->Scw_;

    for (std::set<KeyFrame*>::const_iterator sit = kf_connections.begin(),
                                             sit_end = kf_connections.end();
         sit != sit_end; ++sit) {
      KeyFrame* kf2 = *sit;
      long unsigned int kf2_id = kf2->mnId;
      // Add edges with enough covisibility and all edges connected with
      // pCurKF or pLoopKF
      if ((kf1_id != pCurKF->mnId || kf2_id != pLoopKF->mnId) &&
          kf1->GetWeight(kf2) < min_feat)
        continue;
      // Compute relative pose, in fact these are corrected ones
      Sim3 S2w = kf2->Scw_;
      Sim3 S12 = S1w * S2w.inverse();
      // Create cost function object
      CostFunctionPoseGraphSE3* costfunction =
          new CostFunctionPoseGraphSE3(S12.t, S12.r);
      costfunction->SetInformationMatrix(info_matrix);
      // Add residual block
      ceres_problem.AddResidualBlock(
          costfunction, lossfunction, kf1->Scw_opt_.t.data(),
          kf1->Scw_opt_.r.coeffs().data(), kf2->Scw_opt_.t.data(),
          kf2->Scw_opt_.r.coeffs().data());
      // Set SO3 manifold
      ceres_problem.SetManifold(kf1->Scw_opt_.r.coeffs().data(), so3_manifold);
      ceres_problem.SetManifold(kf2->Scw_opt_.r.coeffs().data(), so3_manifold);
      inserted_edges.insert(
          std::make_pair(min(kf1_id, kf2_id), max(kf1_id, kf2_id)));
    }
  }

  // Add essential graph edges from covisibility graph and spanning tree
  for (size_t i = 0, imax = vec_all_kf.size(); i < imax; ++i) {
    KeyFrame* kf1 = vec_all_kf[i];
    // long unsigned int kf1_id = kf1->mnId;
    Sim3 S1w;
    LoopClosing::KeyFrameAndPose::const_iterator pose_it =
        NonCorrectedSim3.find(kf1);
    if (pose_it != NonCorrectedSim3.end()) {
      S1w = pose_it->second;
    } else {
      S1w = kf1->Scw_;
    }

    // Spanning tree edge
    KeyFrame* parent_kf = kf1->GetParent();
    if (parent_kf) {
      Sim3 Spw;
      LoopClosing::KeyFrameAndPose::const_iterator pose_it_p =
          NonCorrectedSim3.find(parent_kf);
      if (pose_it_p != NonCorrectedSim3.end()) {
        Spw = pose_it_p->second;
      } else {
        Spw = parent_kf->Scw_;
      }
      Sim3 S1p = S1w * Spw.inverse();
      // Create cost function object
      CostFunctionPoseGraphSE3* costfunction =
          new CostFunctionPoseGraphSE3(S1p.t, S1p.r);
      costfunction->SetInformationMatrix(info_matrix);
      // Add residual block
      ceres_problem.AddResidualBlock(
          costfunction, lossfunction, kf1->Scw_opt_.t.data(),
          kf1->Scw_opt_.r.coeffs().data(), parent_kf->Scw_opt_.t.data(),
          parent_kf->Scw_opt_.r.coeffs().data());
      // Set SO3 manifold
      ceres_problem.SetManifold(kf1->Scw_opt_.r.coeffs().data(), so3_manifold);
      ceres_problem.SetManifold(parent_kf->Scw_opt_.r.coeffs().data(),
                                so3_manifold);
    }

    // Loop edges
    const set<KeyFrame*> loop_kfs = kf1->GetLoopEdges();
    for (set<KeyFrame*>::const_iterator it_kf2 = loop_kfs.begin(),
                                        it_kf2_end = loop_kfs.end();
         it_kf2 != it_kf2_end; it_kf2++) {
      KeyFrame* kf2 = *it_kf2;
      if (kf2->mnId < kf1->mnId) {
        Sim3 S2w;
        LoopClosing::KeyFrameAndPose::const_iterator pose_it_2 =
            NonCorrectedSim3.find(kf2);
        if (pose_it_2 != NonCorrectedSim3.end()) {
          S2w = pose_it_2->second;
        } else {
          S2w = kf2->Scw_;
        }
        Sim3 S12 = S1w * S2w.inverse();
        // Create cost function object
        CostFunctionPoseGraphSE3* costfunction =
            new CostFunctionPoseGraphSE3(S12.t, S12.r);
        costfunction->SetInformationMatrix(info_matrix);
        // Add residual block
        ceres_problem.AddResidualBlock(
            costfunction, lossfunction, kf1->Scw_opt_.t.data(),
            kf1->Scw_opt_.r.coeffs().data(), kf2->Scw_opt_.t.data(),
            kf2->Scw_opt_.r.coeffs().data());
        // Set SO3 manifold
        ceres_problem.SetManifold(kf1->Scw_opt_.r.coeffs().data(),
                                  so3_manifold);
        ceres_problem.SetManifold(kf2->Scw_opt_.r.coeffs().data(),
                                  so3_manifold);
      }
    }

    // Covisibility graph edges
    vector<KeyFrame*> connected_kfs = kf1->GetCovisiblesByWeight(min_feat);
    for (size_t i = 0, imax = connected_kfs.size(); i < imax; ++i) {
      KeyFrame* kf2 = connected_kfs[i];
      if (kf2 && kf2 != parent_kf && !kf1->hasChild(kf2) &&
          !loop_kfs.count(kf2)) {
        if (!kf2->isBad() && kf2->mnId < kf1->mnId) {
          if (inserted_edges.count(
                  std::make_pair(std::min(kf1->mnId, kf2->mnId),
                                 std::max(kf1->mnId, kf2->mnId))))
            continue;
          Sim3 S2w;
          LoopClosing::KeyFrameAndPose::const_iterator pose_it_2 =
              NonCorrectedSim3.find(kf2);
          if (pose_it_2 != NonCorrectedSim3.end()) {
            S2w = pose_it_2->second;
          } else {
            S2w = kf2->Scw_;
          }
          Sim3 S12 = S1w * S2w.inverse();
          // Create cost function object
          CostFunctionPoseGraphSE3* costfunction =
              new CostFunctionPoseGraphSE3(S12.t, S12.r);
          costfunction->SetInformationMatrix(info_matrix);
          // Add residual block
          ceres_problem.AddResidualBlock(
              costfunction, lossfunction, kf1->Scw_opt_.t.data(),
              kf1->Scw_opt_.r.coeffs().data(), kf2->Scw_opt_.t.data(),
              kf2->Scw_opt_.r.coeffs().data());
          // Set SO3 manifold
          ceres_problem.SetManifold(kf1->Scw_opt_.r.coeffs().data(),
                                    so3_manifold);
          ceres_problem.SetManifold(kf2->Scw_opt_.r.coeffs().data(),
                                    so3_manifold);
        }
      }
    }
  }

  // std::cout << "NumResidualBlocks: " << ceres_problem.NumResidualBlocks()
  //           << std::endl;
  // Set LoopKF as fixed
  ceres_problem.SetParameterBlockConstant(pLoopKF->Scw_opt_.r.coeffs().data());
  ceres_problem.SetParameterBlockConstant(pLoopKF->Scw_opt_.t.data());

  ceres::Solve(options, &ceres_problem, &summary);

  // std::cout << summary.BriefReport()
  //           << "Used time: " << summary.total_time_in_seconds << " sec "
  //           << std::endl;

  std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);
  /// Recovery KF data
  for (size_t i = 0, imax = vec_all_kf.size(); i < imax; ++i) {
    KeyFrame* kf = vec_all_kf[i];
    if (kf->isBad()) continue;
    Sim3& Scw = kf->Scw_opt_;
    Eigen::Matrix3d Rcw = Scw.r.matrix();
    Eigen::Vector3d tcw = Scw.t;

    Eigen::Isometry3d Tcw_eigen(Rcw);
    Tcw_eigen.pretranslate(tcw);
    cv::Mat Tcw(4, 4, CV_32F);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        Tcw.at<float>(i, j) = Tcw_eigen(i, j);
      }
    }
    kf->SetPose(Tcw);
  }

  /// Recovery MP data
  for (size_t i = 0, imax = vec_all_mp.size(); i < imax; ++i) {
    MapPoint* mp = vec_all_mp[i];
    if (mp->isBad()) continue;
    Sim3 Srw_old;
    Sim3 Srw_new;
    if (mp->mnCorrectedByKF == pCurKF->mnId) {
      Srw_old = mp->mnCorrectedReferencePtr->Scw_;
      Srw_new = mp->mnCorrectedReferencePtr->Scw_opt_;
    } else {
      Srw_old = mp->GetReferenceKeyFrame()->Scw_;
      Srw_new = mp->GetReferenceKeyFrame()->Scw_opt_;
    }
    cv::Mat twp_cv = mp->GetWorldPos();
    Eigen::Vector3d twp_old(twp_cv.at<float>(0), twp_cv.at<float>(01),
                            twp_cv.at<float>(2));
    Eigen::Vector3d twp_new = Srw_new.inverse().map(Srw_old.map(twp_old));
    cv::Mat twp_new_cv(3, 1, CV_32F);
    twp_new_cv.at<float>(0) = twp_new(0);
    twp_new_cv.at<float>(1) = twp_new(1);
    twp_new_cv.at<float>(2) = twp_new(2);
    mp->SetWorldPos(twp_new_cv);
    mp->UpdateNormalAndDepth();
  }
}

// /// This function need to be refined
// /// Should use different variables with PoseBA and LocalBA
// void CeresOptimizer::GlobalBundleAdjustment(Map* pMap, int nIterations,
//                                             const unsigned long nLoopKF,
//                                             const bool bRobust) {
//   // std::cout << "Ceres GlobalBundleAdjustment" << std::endl;
//   /// Prepare optimization variable
//   std::vector<KeyFrame*> vec_keyframes = pMap->GetAllKeyFrames();
//   std::vector<MapPoint*> vec_mappoints = pMap->GetAllMapPoints();

//   long unsigned int max_kf_idx = 0;
//   for (size_t i = 0, imax = vec_keyframes.size(); i < imax; ++i) {
//     KeyFrame* kf = vec_keyframes[i];
//     kf->UpdateEigenPose();
//     kf->InitOptVariable();
//     if (kf->mnId > max_kf_idx) max_kf_idx = kf->mnId;
//   }
//   for (size_t i = 0, imax = vec_mappoints.size(); i < imax; ++i) {
//     MapPoint* mp = vec_mappoints[i];
//     mp->UpdateEigenPose();
//     mp->InitOptVariable();
//   }

//   /// Setup Ceres
//   ceres::Problem::Options problem_options;
//   ceres::Problem ceres_problem(problem_options);
//   ceres::Solver::Options options;
//   options.minimizer_progress_to_stdout = false;
//   options.max_num_iterations = nIterations;
//   options.linear_solver_type = ceres::SPARSE_SCHUR;
//   // TODO: find better options!
//   ceres::Solver::Summary summary;

//   // LossFunction thresholds
//   const double delta_mono = sqrt(5.991);
//   const double delta_stereo = sqrt(7.815);
//   ceres::LossFunction* lossfunction_mono = nullptr;
//   ceres::LossFunction* lossfunction_stereo = nullptr;
//   if (bRobust) {
//     lossfunction_mono = new ceres::HuberLoss(delta_mono);
//     lossfunction_stereo = new ceres::HuberLoss(delta_stereo);
//   }
//   QuaternionManifoldIdentity* quaternion_manifold =
//       new QuaternionManifoldIdentity;

//   for (size_t i = 0, imax = vec_mappoints.size(); i < imax; ++i) {
//     MapPoint* mp = vec_mappoints[i];
//     if (!mp || mp->isBad()) continue;
//     const std::map<ORB_SLAM2::KeyFrame*, size_t> observations =
//         mp->GetObservations();
//     for (std::map<ORB_SLAM2::KeyFrame*, size_t>::const_iterator
//              it = observations.begin(),
//              it_end = observations.end();
//          it != it_end; ++it) {
//       KeyFrame* kf = it->first;
//       if (kf->isBad() || kf->mnId > max_kf_idx) continue;
//       int kp_idx = it->second;
//       const cv::KeyPoint& kp = kf->mvKeysUn[kp_idx];
//       double u_right = kf->mvuRight[kp_idx];
//       if (u_right < 0) {
//         // Mono observation
//         // Create cost function object
//         CostFunctionSE3ProjectionMono* cost_ptr =
//             new CostFunctionSE3ProjectionMono(
//                 kf->fx, kf->fy, kf->cx, kf->cy,
//                 Eigen::Vector2d(kp.pt.x, kp.pt.y));
//         // Information matrix
//         const float inv_sigma2 = kf->mvInvLevelSigma2[kp.octave];
//         cost_ptr->SetInformationMatrix(Eigen::Matrix2d::Identity() *
//                                        sqrt(inv_sigma2));
//         // Add residual to problem
//         ceres_problem.AddResidualBlock(
//             cost_ptr, lossfunction_mono, kf->tcw_opt_.data(),
//             kf->qcw_opt_.coeffs().data(), mp->twp_opt_.data());
//         // Set manifold
//         ceres_problem.SetManifold(kf->qcw_opt_.coeffs().data(),
//                                   quaternion_manifold);
//       } else {
//         // Stereo observation
//         // Create cost function object
//         CostFunctionSE3ProjectionStereo* cost_ptr =
//             new CostFunctionSE3ProjectionStereo(
//                 kf->fx, kf->fy, kf->cx, kf->cy, kf->mbf,
//                 Eigen::Vector3d(kp.pt.x, kp.pt.y, u_right));
//         // Information matrix
//         const float inv_sigma2 = kf->mvInvLevelSigma2[kp.octave];
//         cost_ptr->SetInformationMatrix(Eigen::Matrix3d::Identity() *
//                                        sqrt(inv_sigma2));
//         // Add residual to problem
//         ceres_problem.AddResidualBlock(
//             cost_ptr, lossfunction_stereo, kf->tcw_opt_.data(),
//             kf->qcw_opt_.coeffs().data(), mp->twp_opt_.data());
//         // Set manifold
//         ceres_problem.SetManifold(kf->qcw_opt_.coeffs().data(),
//                                   quaternion_manifold);
//       }
//       if (kf->mnId == 0) {
//         ceres_problem.SetParameterBlockConstant(kf->tcw_opt_.data());
//         ceres_problem.SetParameterBlockConstant(kf->qcw_opt_.coeffs().data());
//       }
//     }
//   }

//   ceres::Solve(options, &ceres_problem, &summary);

//   /// Recovery KF data
//   for (size_t i = 0, imax = vec_keyframes.size(); i < imax; ++i) {
//     KeyFrame* kf = vec_keyframes[i];
//     if (kf->isBad()) continue;
//     kf->SetFromOptResult();
//     Eigen::Isometry3d Tcw_eigen(kf->qcw_);
//     Tcw_eigen.pretranslate(kf->tcw_);
//     cv::Mat Tcw(4, 4, CV_32F);
//     for (size_t i = 0; i < 4; ++i) {
//       for (size_t j = 0; j < 4; ++j) {
//         Tcw.at<float>(i, j) = Tcw_eigen(i, j);
//       }
//     }
//     if (nLoopKF == 0) {
//       kf->SetPose(Tcw);
//     } else {
//       kf->mTcwGBA.create(4, 4, CV_32F);
//       Tcw.copyTo(kf->mTcwGBA);
//       kf->mnBAGlobalForKF = nLoopKF;
//     }
//   }

//   /// Recovery MP data
//   for (size_t i = 0, imax = vec_mappoints.size(); i < imax; ++i) {
//     MapPoint* mp = vec_mappoints[i];
//     if (mp->isBad()) continue;
//     mp->SetFromOptResult();
//     if (nLoopKF == 0) {
//       mp->UpdatePoseFromEigen();
//       mp->UpdateNormalAndDepth();
//     } else {
//       mp->mPosGBA.create(3, 1, CV_32F);
//       for (size_t i = 0; i < 3; ++i) {
//         mp->mPosGBA.at<float>(i, 0) = mp->twp_[i];
//       }
//       mp->mnBAGlobalForKF = nLoopKF;
//     }
//   }
// }

void CeresOptimizer::GlobalBundleAdjustment(Map* pMap, int nIterations,
                                            const unsigned long nLoopKF,
                                            const bool bRobust) {
  // std::cout << "Ceres GlobalBundleAdjustment" << std::endl;
  /// Prepare optimization variable
  std::vector<KeyFrame*> vec_keyframes = pMap->GetAllKeyFrames();
  std::vector<MapPoint*> vec_mappoints = pMap->GetAllMapPoints();

  long unsigned int max_kf_idx = 0;
  for (size_t i = 0, imax = vec_keyframes.size(); i < imax; ++i) {
    KeyFrame* kf = vec_keyframes[i];
    cv::Mat Tcw = kf->GetPose();
    Eigen::Isometry3d Tcw_eigen;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        Tcw_eigen(i, j) = Tcw.at<float>(i, j);
      }
    }
    kf->tcw_gba_opt = Tcw_eigen.translation();
    kf->qcw_gba_opt = Eigen::Quaterniond(Tcw_eigen.rotation());
    if (kf->mnId > max_kf_idx) max_kf_idx = kf->mnId;
  }
  for (size_t i = 0, imax = vec_mappoints.size(); i < imax; ++i) {
    MapPoint* mp = vec_mappoints[i];
    cv::Mat tp = mp->GetWorldPos();
    for (size_t i = 0; i < 3; ++i) {
      mp->twp_gba_opt[i] = tp.at<float>(i, 0);
    }
  }

  /// Setup Ceres
  ceres::Problem::Options problem_options;
  ceres::Problem ceres_problem(problem_options);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = nIterations;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  // options.linear_solver_ordering_type
  // TODO: find better options!
  ceres::Solver::Summary summary;

  // LossFunction thresholds
  const double delta_mono = sqrt(5.991);
  const double delta_stereo = sqrt(7.815);
  ceres::LossFunction* lossfunction_mono = nullptr;
  ceres::LossFunction* lossfunction_stereo = nullptr;
  if (bRobust) {
    lossfunction_mono = new ceres::HuberLoss(delta_mono);
    lossfunction_stereo = new ceres::HuberLoss(delta_stereo);
  }
  QuaternionManifoldIdentity* quaternion_manifold =
      new QuaternionManifoldIdentity;

  // Add residuals to problem
  for (size_t i = 0, imax = vec_mappoints.size(); i < imax; ++i) {
    MapPoint* mp = vec_mappoints[i];
    if (!mp || mp->isBad()) continue;
    const std::map<ORB_SLAM2::KeyFrame*, size_t> observations =
        mp->GetObservations();
    for (std::map<ORB_SLAM2::KeyFrame*, size_t>::const_iterator
             it = observations.begin(),
             it_end = observations.end();
         it != it_end; ++it) {
      KeyFrame* kf = it->first;
      if (kf->isBad() || kf->mnId > max_kf_idx) continue;
      int kp_idx = it->second;
      const cv::KeyPoint& kp = kf->mvKeysUn[kp_idx];
      double u_right = kf->mvuRight[kp_idx];
      if (u_right < 0) {
        // Mono observation
        // Create cost function object
        CostFunctionSE3ProjectionMono* cost_ptr =
            new CostFunctionSE3ProjectionMono(
                kf->fx, kf->fy, kf->cx, kf->cy,
                Eigen::Vector2d(kp.pt.x, kp.pt.y));
        // Information matrix
        const float inv_sigma2 = kf->mvInvLevelSigma2[kp.octave];
        cost_ptr->SetInformationMatrix(Eigen::Matrix2d::Identity() *
                                       sqrt(inv_sigma2));
        // Add residual to problem
        ceres_problem.AddResidualBlock(
            cost_ptr, lossfunction_mono, kf->tcw_gba_opt.data(),
            kf->qcw_gba_opt.coeffs().data(), mp->twp_gba_opt.data());
        // Set manifold
        ceres_problem.SetManifold(kf->qcw_gba_opt.coeffs().data(),
                                  quaternion_manifold);
      } else {
        // Stereo observation
        // Create cost function object
        CostFunctionSE3ProjectionStereo* cost_ptr =
            new CostFunctionSE3ProjectionStereo(
                kf->fx, kf->fy, kf->cx, kf->cy, kf->mbf,
                Eigen::Vector3d(kp.pt.x, kp.pt.y, u_right));
        // Information matrix
        const float inv_sigma2 = kf->mvInvLevelSigma2[kp.octave];
        cost_ptr->SetInformationMatrix(Eigen::Matrix3d::Identity() *
                                       sqrt(inv_sigma2));
        // Add residual to problem
        ceres_problem.AddResidualBlock(
            cost_ptr, lossfunction_stereo, kf->tcw_gba_opt.data(),
            kf->qcw_gba_opt.coeffs().data(), mp->twp_gba_opt.data());
        // Set manifold
        ceres_problem.SetManifold(kf->qcw_gba_opt.coeffs().data(),
                                  quaternion_manifold);
      }
      if (kf->mnId == 0) {
        ceres_problem.SetParameterBlockConstant(kf->tcw_gba_opt.data());
        ceres_problem.SetParameterBlockConstant(
            kf->qcw_gba_opt.coeffs().data());
      }
    }
  }

  ceres::Solve(options, &ceres_problem, &summary);

  /// Recovery KF data
  for (size_t i = 0, imax = vec_keyframes.size(); i < imax; ++i) {
    KeyFrame* kf = vec_keyframes[i];
    if (kf->isBad()) continue;
    Eigen::Isometry3d Tcw_eigen(kf->qcw_gba_opt);
    Tcw_eigen.pretranslate(kf->tcw_gba_opt);
    cv::Mat Tcw(4, 4, CV_32F);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        Tcw.at<float>(i, j) = Tcw_eigen(i, j);
      }
    }
    if (nLoopKF == 0) {
      kf->SetPose(Tcw);
    } else {
      kf->mTcwGBA.create(4, 4, CV_32F);
      Tcw.copyTo(kf->mTcwGBA);
      kf->mnBAGlobalForKF = nLoopKF;
    }
  }

  /// Recovery MP data
  for (size_t i = 0, imax = vec_mappoints.size(); i < imax; ++i) {
    MapPoint* mp = vec_mappoints[i];
    if (mp->isBad()) continue;
    if (nLoopKF == 0) {
      cv::Mat pw(3, 1, CV_32F);
      for (size_t i = 0; i < 3; ++i) {
        pw.at<float>(i, 0) = mp->twp_gba_opt[i];
      }
      mp->UpdateNormalAndDepth();
    } else {
      mp->mPosGBA.create(3, 1, CV_32F);
      for (size_t i = 0; i < 3; ++i) {
        mp->mPosGBA.at<float>(i, 0) = mp->twp_gba_opt[i];
      }
      mp->mnBAGlobalForKF = nLoopKF;
    }
  }
}

}  // namespace ORB_SLAM2
