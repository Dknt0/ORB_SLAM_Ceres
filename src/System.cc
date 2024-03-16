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

#include "System.h"

#include <pangolin/pangolin.h>

#include <iomanip>
#include <thread>

#include "Converter.h"

namespace ORB_SLAM2 {

/// @brief 系统构造函数
/// @param strVocFile 词袋路径
/// @param strSettingsFile 配置文件路径
/// @param sensor 传感器类型
/// @param bUseViewer 是否使用 Viewer
System::System(const string &strVocFile, const string &strSettingsFile,
               const eSensor sensor, const bool bUseViewer)
    : mSensor(sensor),
      mpViewer(static_cast<Viewer *>(NULL)),
      mbReset(false),
      mbActivateLocalizationMode(false),
      mbDeactivateLocalizationMode(false) {
  // Output welcome message
  cout << endl
       << "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of "
          "Zaragoza."
       << endl
       << "This program comes with ABSOLUTELY NO WARRANTY;" << endl
       << "This is free software, and you are welcome to redistribute it"
       << endl
       << "under certain conditions. See LICENSE.txt." << endl
       << endl;

  cout << "Input sensor was set to: ";

  if (mSensor == MONOCULAR)
    cout << "Monocular" << endl;
  else if (mSensor == STEREO)
    cout << "Stereo" << endl;
  else if (mSensor == RGBD)
    cout << "RGB-D" << endl;

  // Check settings file
  cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    cerr << "Failed to open settings file at: " << strSettingsFile << endl;
    exit(-1);
  }

  // Load ORB Vocabulary
  cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

  mpVocabulary = new ORBVocabulary();
  bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
  if (!bVocLoad) {
    cerr << "Wrong path to vocabulary. " << endl;
    cerr << "Falied to open at: " << strVocFile << endl;
    exit(-1);
  }
  cout << "Vocabulary loaded!" << endl << endl;

  mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
  mpMap = new Map();

  // FrameDrawer 和 MapDrawer 是直接在 Pangolin 中绘图的
  // Create Drawers. These are used by the Viewer
  mpFrameDrawer = new FrameDrawer(mpMap);
  mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

  // 跟踪线程，主线程
  // Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this
  // constructor)
  mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                           mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

  // Initialize the Local Mapping thread and launch
  mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);
  mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

  // Initialize the Loop Closing thread and launch
  mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary,
                                 mSensor != MONOCULAR);
  mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

  // Initialize the Viewer thread and launch
  if (bUseViewer) {
    mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker,
                          strSettingsFile);
    mptViewer = new thread(&Viewer::Run, mpViewer);
    mpTracker->SetViewer(mpViewer);
  }

  // 三个线程之间是互相联系的
  // Set pointers between threads
  mpTracker->SetLocalMapper(mpLocalMapper);
  mpTracker->SetLoopClosing(mpLoopCloser);

  mpLocalMapper->SetTracker(mpTracker);
  mpLocalMapper->SetLoopCloser(mpLoopCloser);

  mpLoopCloser->SetTracker(mpTracker);
  mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

/// @brief 追踪双目图像  先检查标志位，设置对应模式。之后将图像输入到
/// Tracking，并更新地图点和特征点
/// @param imLeft
/// @param imRight
/// @param timestamp
/// @return
cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                            const double &timestamp) {
  if (mSensor != STEREO) {
    cerr << "ERROR: you called TrackStereo but input sensor was not set to "
            "STEREO."
         << endl;
    exit(-1);
  }

  /* 1.重定位模式检查 */
  // 这里为什么要用双标志?
  // Check mode change
  {
    unique_lock<mutex> lock(mMutexMode);
    if (mbActivateLocalizationMode) {
      // 暂停局部地图管理
      mpLocalMapper->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!mpLocalMapper->isStopped()) {
        usleep(1000);
      }

      mpTracker->InformOnlyTracking(true);
      mbActivateLocalizationMode = false;
    }
    if (mbDeactivateLocalizationMode) {
      mpTracker->InformOnlyTracking(false);
      // 开启局部地图管理
      mpLocalMapper->Release();
      mbDeactivateLocalizationMode = false;
    }
  }

  /* 2.退出检查 */
  // Check reset
  {
    unique_lock<mutex> lock(mMutexReset);
    if (mbReset) {
      mpTracker->Reset();
      mbReset = false;
    }
  }

  /* 3.追踪双目图像  调用 Tracking::GrabImageStereo */

  // 没有矫畸变，ORB-SLAM 中的矫畸变是在特征提取之后进行的
  cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

  /* 4.同步 Tracking 中当前帧信息 */
  // 将 Tracker 中的地图点和特征点更新到了系统
  unique_lock<mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return Tcw;
}

/// @brief 追踪深度图  先检查标志位，设置对应模式。之后将图像输入到
/// Tracking，并更新地图点和特征点
/// @param im
/// @param depthmap
/// @param timestamp
/// @return
cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap,
                          const double &timestamp) {
  if (mSensor != RGBD) {
    cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD."
         << endl;
    exit(-1);
  }

  // Check mode change
  {
    unique_lock<mutex> lock(mMutexMode);
    if (mbActivateLocalizationMode) {
      mpLocalMapper->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!mpLocalMapper->isStopped()) {
        usleep(1000);
      }

      mpTracker->InformOnlyTracking(true);
      mbActivateLocalizationMode = false;
    }
    if (mbDeactivateLocalizationMode) {
      mpTracker->InformOnlyTracking(false);
      mpLocalMapper->Release();
      mbDeactivateLocalizationMode = false;
    }
  }

  // Check reset
  {
    unique_lock<mutex> lock(mMutexReset);
    if (mbReset) {
      mpTracker->Reset();
      mbReset = false;
    }
  }

  cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp);

  unique_lock<mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return Tcw;
}

/// @brief 追踪单目图像  先检查标志位，设置对应模式。之后将图像输入到
/// Tracking，并更新地图点和特征点
/// @param im
/// @param timestamp
/// @return
cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp) {
  if (mSensor != MONOCULAR) {
    cerr << "ERROR: you called TrackMonocular but input sensor was not set to "
            "Monocular."
         << endl;
    exit(-1);
  }

  // Check mode change
  {
    unique_lock<mutex> lock(mMutexMode);
    if (mbActivateLocalizationMode) {
      mpLocalMapper->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!mpLocalMapper->isStopped()) {
        usleep(1000);
      }

      mpTracker->InformOnlyTracking(true);
      mbActivateLocalizationMode = false;
    }
    if (mbDeactivateLocalizationMode) {
      mpTracker->InformOnlyTracking(false);
      mpLocalMapper->Release();
      mbDeactivateLocalizationMode = false;
    }
  }

  // Check reset
  {
    unique_lock<mutex> lock(mMutexReset);
    if (mbReset) {
      mpTracker->Reset();
      mbReset = false;
    }
  }

  // 没有矫畸变？
  cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

  unique_lock<mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

  return Tcw;
}

/// @brief 启动纯定位模式
void System::ActivateLocalizationMode() {
  unique_lock<mutex> lock(mMutexMode);
  mbActivateLocalizationMode = true;
}

/// @brief 关闭纯定位模式
void System::DeactivateLocalizationMode() {
  unique_lock<mutex> lock(mMutexMode);
  mbDeactivateLocalizationMode = true;
}

/// @brief 距离上次调用地图是否因为闭环、全局BA发生大改变
/// @return
bool System::MapChanged() {
  static int n = 0;  // 静态变量
  int curn = mpMap->GetLastBigChangeIdx();
  if (n < curn) {
    n = curn;
    return true;
  } else
    return false;
}

/// @brief 重置系统
void System::Reset() {
  unique_lock<mutex> lock(mMutexReset);
  mbReset = true;
}

/// @brief 关闭系统  保存轨迹之前必须先调用该函数
void System::Shutdown() {
  mpLocalMapper->RequestFinish();
  mpLoopCloser->RequestFinish();
  if (mpViewer) {
    mpViewer->RequestFinish();
    while (!mpViewer->isFinished()) usleep(5000);
  }

  // Wait until all thread have effectively stopped
  while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() ||
         mpLoopCloser->isRunningGBA()) {
    usleep(5000);
  }

  // ?
  if (mpViewer) pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

/// @brief 按照 TUM RGB-D 数据集的格式保存相机轨迹
/// Only for stereo and RGB-D. This method does not work for monocular.
/// Call first Shutdown()
/// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
///
/// @param filename
void System::SaveTrajectoryTUM(const string &filename) {
  cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
  if (mSensor == MONOCULAR) {
    cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
    return;
  }

  vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  // Frame pose is stored relative to its reference keyframe (which is optimized
  // by BA and pose graph). We need to get first the keyframe pose and then
  // concatenate the relative transformation. Frames not localized (tracking
  // failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and
  // a flag which is true when tracking failed (lbL).
  list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
  list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  list<bool>::iterator lbL = mpTracker->mlbLost.begin();
  for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
                               lend = mpTracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++, lbL++) {
    if (*lbL) continue;

    KeyFrame *pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    // If the reference keyframe was culled, traverse the spanning tree to get a
    // suitable keyframe.
    while (pKF->isBad()) {
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    vector<float> q = Converter::toQuaternion(Rwc);

    f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0)
      << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0]
      << " " << q[1] << " " << q[2] << " " << q[3] << endl;
  }
  f.close();
  cout << endl << "trajectory saved!" << endl;
}

/// @brief 按照 TUM RGB-D 数据集的格式保存关键帧的相机位姿
/// This method works for all sensor input.
/// Call first Shutdown()
/// See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
///
/// @param filename
void System::SaveKeyFrameTrajectoryTUM(const string &filename) {
  cout << endl
       << "Saving keyframe trajectory to " << filename << " ..." << endl;

  vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  // cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame *pKF = vpKFs[i];

    // pKF->SetPose(pKF->GetPose()*Two);

    if (pKF->isBad()) continue;

    cv::Mat R = pKF->GetRotation().t();
    vector<float> q = Converter::toQuaternion(R);
    cv::Mat t = pKF->GetCameraCenter();
    f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " "
      << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " "
      << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
  }

  f.close();
  cout << endl << "trajectory saved!" << endl;
}

/// @brief 按照 KITTI 数据集的格式保存相机轨迹
/// Only for stereo and RGB-D. This method does not work for monocular.
/// Call first Shutdown()
/// See format details at:
/// http://www.cvlibs.net/datasets/kitti/eval_odometry.php
///
/// @param filename
void System::SaveTrajectoryKITTI(const string &filename) {
  cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
  if (mSensor == MONOCULAR) {
    cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
    return;
  }

  vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
  sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat Two = vpKFs[0]->GetPoseInverse();

  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  // Frame pose is stored relative to its reference keyframe (which is optimized
  // by BA and pose graph). We need to get first the keyframe pose and then
  // concatenate the relative transformation. Frames not localized (tracking
  // failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and
  // a flag which is true when tracking failed (lbL).
  list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
  list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
                               lend = mpTracker->mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lT++) {
    ORB_SLAM2::KeyFrame *pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

    while (pKF->isBad()) {
      //  cout << "bad parent" << endl;
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * Two;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

    f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1)
      << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << " "
      << Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " "
      << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " "
      << Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " "
      << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;
  }
  f.close();
  cout << endl << "trajectory saved!" << endl;
}

/// @brief 获取追踪状态
/// @return
int System::GetTrackingState() {
  unique_lock<mutex> lock(mMutexState);
  return mTrackingState;
}

/// @brief 获取追踪到的地图点
/// @return
vector<MapPoint *> System::GetTrackedMapPoints() {
  unique_lock<mutex> lock(mMutexState);
  return mTrackedMapPoints;
}

/// @brief 获取校畸变后的特征点
/// @return
vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
  unique_lock<mutex> lock(mMutexState);
  return mTrackedKeyPointsUn;
}

}  // namespace ORB_SLAM2
