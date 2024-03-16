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

#include "Tracking.h"

#include <opencv2/imgproc/types_c.h>

#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "Converter.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "CeresOptimizer.h"
#include "PnPsolver.h"

using namespace std;

namespace ORB_SLAM2 {
/// @brief 追踪器构造函数
/// @param pSys 系统类指针
/// @param pVoc 视觉字典
/// @param pFrameDrawer 帧绘制器指针
/// @param pMapDrawer 地图绘制器指针
/// @param pMap 地图
/// @param pKFDB 关键帧数据库
/// @param strSettingPath 配置文件路径
/// @param sensor 传感器类型
Tracking::Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer,
                   MapDrawer* pMapDrawer, Map* pMap, KeyFrameDatabase* pKFDB,
                   const string& strSettingPath, const int sensor)
    : mState(NO_IMAGES_YET),
      mSensor(sensor),
      mbOnlyTracking(false),
      mbVO(false),
      mpSystem(pSys),
      mpViewer(NULL),
      mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer),
      mpInitializer(static_cast<Initializer*>(NULL)),
      mpORBVocabulary(pVoc),
      mpKeyFrameDB(pKFDB),
      mpMap(pMap),
      mnLastRelocFrameId(0) {
  /* 1.从配置文件读取参数 */
  // Load camera parameters from settings file
  // OpenCV 提供的 yaml 文件解析器
  // 相机内参
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  // 畸变参数
  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  // baseline times fx
  mbf = fSettings["Camera.bf"];

  float fps = fSettings["Camera.fps"];
  if (fps == 0) fps = 30;

  // Max/Min Frames to insert keyframes and to check relocalisation
  mMinFrames = 0;
  mMaxFrames = fps;

  cout << endl << "Camera Parameters: " << endl;
  cout << "- fx: " << fx << endl;
  cout << "- fy: " << fy << endl;
  cout << "- cx: " << cx << endl;
  cout << "- cy: " << cy << endl;
  cout << "- k1: " << DistCoef.at<float>(0) << endl;
  cout << "- k2: " << DistCoef.at<float>(1) << endl;
  if (DistCoef.rows == 5) cout << "- k3: " << DistCoef.at<float>(4) << endl;
  cout << "- p1: " << DistCoef.at<float>(2) << endl;
  cout << "- p2: " << DistCoef.at<float>(3) << endl;
  cout << "- fps: " << fps << endl;

  int nRGB = fSettings["Camera.RGB"];
  mbRGB = nRGB;

  if (mbRGB)
    cout << "- color order: RGB (ignored if grayscale)" << endl;
  else
    cout << "- color order: BGR (ignored if grayscale)" << endl;

  // Load ORB parameters
  int nFeatures = fSettings["ORBextractor.nFeatures"];  // 单帧提取 ORB 特征数
  float fScaleFactor = fSettings["ORBextractor.scaleFactor"];  // 金字塔层间比例
  int nLevels = fSettings["ORBextractor.nLevels"];       // 金字塔层数
  int fIniThFAST = fSettings["ORBextractor.iniThFAST"];  // ORB 初始阈值
  int fMinThFAST = fSettings["ORBextractor.minThFAST"];  // ORB 最小阈值

  /* 2.创建提取器对象 */
  // 创建 ORB 提取器
  mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                        fIniThFAST, fMinThFAST);

  if (sensor == System::STEREO)
    mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                           fIniThFAST, fMinThFAST);

  // 单目初始化提取器提取了设定值双倍的特征
  if (sensor == System::MONOCULAR)
    mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels,
                                         fIniThFAST, fMinThFAST);

  cout << endl << "ORB Extractor Parameters: " << endl;
  cout << "- Number of Features: " << nFeatures << endl;
  cout << "- Scale Levels: " << nLevels << endl;
  cout << "- Scale Factor: " << fScaleFactor << endl;
  cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
  cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

  // 设置远点阈值
  if (sensor == System::STEREO || sensor == System::RGBD) {
    mThDepth = mbf * (float)fSettings["ThDepth"] /
               fx;  // 配置文件中 ThDepth 是基线的倍数
    cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
  }

  // DepthMapFactor 指 1m 对应于深度图像中的灰度值
  if (sensor == System::RGBD) {
    mDepthMapFactor = fSettings["DepthMapFactor"];
    if (fabs(mDepthMapFactor) < 1e-5)
      mDepthMapFactor = 1;
    else
      mDepthMapFactor = 1.0f / mDepthMapFactor;
  }
}

/// @brief 设置 LocalMapping 指针
/// @param pLocalMapper
void Tracking::SetLocalMapper(LocalMapping* pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

/// @brief 设置 LoopClosing 指针
/// @param pLoopClosing
void Tracking::SetLoopClosing(LoopClosing* pLoopClosing) {
  mpLoopClosing = pLoopClosing;
}

/// @brief 设置显示器指针
/// @param pViewer
void Tracking::SetViewer(Viewer* pViewer) { mpViewer = pViewer; }

/// @brief 双目捕捉
/// @param imRectLeft 左图
/// @param imRectRight 右图
/// @param timestamp 时间戳
/// @return Tcw
cv::Mat Tracking::GrabImageStereo(const cv::Mat& imRectLeft,
                                  const cv::Mat& imRectRight,
                                  double timestamp) {
  mImGray = imRectLeft;
  cv::Mat imGrayRight = imRectRight;

  // 考虑了各种图像格式
  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
    }
  }

  // 从双目图像构建 Frame
  mCurrentFrame =
      Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft,
            mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

  Track();

  return mCurrentFrame.mTcw.clone();
}

/// @brief 深度捕捉
/// @param imRGB 彩色/灰度图
/// @param imD 深度图
/// @param timestamp 时间戳
/// @return Tcw
cv::Mat Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                                double timestamp) {
  mImGray = imRGB;
  cv::Mat imDepth = imD;

  if (3 == mImGray.channels()) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (4 == mImGray.channels()) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  // 深度转换 uint16 转 float，结果以米为单位
  // 如果深度系数大于 1，且深度图类型不是 float，转换之
  // 用户可以传入 uint16 类型的深度图，或 float
  // 深度图，前者通常需要乘深度系数，后者像素中直接存放深度
  if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
    imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

  // 从深度图像构建 Frame
  mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft,
                        mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

  Track();
  return mCurrentFrame.mTcw.clone();
}

/// @brief 捕捉单目
/// @param im
/// @param timestamp
/// @return
cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im, double timestamp) {
  mImGray = im;

  if (3 == mImGray.channels()) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (4 == mImGray.channels()) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  // 是否已经初始化
  if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) {
    // 从单目图像，由初始化提取器创建 Frame
    mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor,
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
  } else {
    // 从单目图像，由左图提取器创建 Frame
    mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft,
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
  }

  Track();
  return mCurrentFrame.mTcw.clone();
}

/// @brief 追踪
void Tracking::Track() {
  if (mState == NO_IMAGES_YET) mState = NOT_INITIALIZED;
  mLastProcessedState = mState;

  // 地图更新锁
  unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

  /* 检查初始化 */
  if (mState == NOT_INITIALIZED) {
    // 如果初始化未完成，进行初始化
    if (mSensor == System::STEREO || mSensor == System::RGBD)
      StereoInitialization();  // 深度相机看作双目
    else
      MonocularInitialization();  // 单目初始化

    mpFrameDrawer->Update(this);  // 绘制帧
    if (mState != OK) return;
  } else {
    bool bOK;  // 本次追踪成功标志位

    /* 追踪一帧 */
    // Initial camera pose estimation using motion model or relocalization (if
    // tracking is lost)
    if (!mbOnlyTracking) {
      // SLAM 模式
      if (OK == mState) {
        // SLAM 模式，正常运行状态
        // Local Mapping might have changed some MapPoints tracked in last frame
        // 局部地图管理器可能会修改上一帧定位到的地图点
        CheckReplacedInLastFrame();
        // 如果速度信息缺失，或刚刚完成了重定位，根据参考 KF
        // 追踪，否则根据速度模型追踪
        if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
          bOK = TrackReferenceKeyFrame();  // 根据参考关键帧跟踪
        } else {
          bOK = TrackWithMotionModel();  // 根据匀速模型跟踪速度更快
          if (!bOK)
            bOK =
                TrackReferenceKeyFrame();  // 如果速度模型追踪失败，根据参考关键帧跟踪
        }
      } else {
        // SLAM 模式，追踪失败
        bOK = Relocalization();  // 重定位
      }
    } else {
      // 纯定位模式
      if (mState == LOST) {
        // 纯定位模式，追踪失败
        bOK = Relocalization();  // 跟丢就重定位
      } else {
        if (!mbVO) {
          // 纯定位模式，上一帧追踪到足够数量 MP
          // In last frame we tracked enough MapPoints in the map
          if (!mVelocity.empty())
            bOK = TrackWithMotionModel();  // 运动模型追踪
          else
            bOK = TrackReferenceKeyFrame();  // 参考关键帧追踪
        } else {
          // 纯定位模式，上一帧追踪 MP 数量不够
          // In last frame we tracked mainly "visual odometry" points.
          // We compute two camera poses, one from motion model and one doing
          // relocalization. If relocalization is sucessfull we choose that
          // solution, otherwise we retain the "visual odometry" solution.
          // 用于存放计算结果
          bool bOKMM = false;         // 运动模型追踪成功
          bool bOKReloc = false;      // 重定位追踪成功
          vector<MapPoint*> vpMPsMM;  // 当前帧地图点观测
          vector<bool> vbOutMM;       // 当前帧地图点观测成功标志位
          cv::Mat TcwMM;              // 运动模型计算 Tcw
          if (!mVelocity.empty()) {
            bOKMM = TrackWithMotionModel();  // 运动模型追踪
            vpMPsMM = mCurrentFrame.mvpMapPoints;
            vbOutMM = mCurrentFrame.mvbOutlier;
            TcwMM = mCurrentFrame.mTcw.clone();
          }
          bOKReloc = Relocalization();  // 重定位

          // 运动模型成功，重定位失败
          if (bOKMM && !bOKReloc) {
            // 使用运动模型的追踪结果
            mCurrentFrame.SetPose(TcwMM);
            mCurrentFrame.mvpMapPoints = vpMPsMM;
            mCurrentFrame.mvbOutlier = vbOutMM;

            // 如果追踪失败，更新 MP 检测次数？
            if (mbVO) {
              for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvpMapPoints[i] &&
                    !mCurrentFrame.mvbOutlier[i])
                  mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
              }
            }
          } else if (bOKReloc) {
            mbVO = false;  // 重定位成功，恢复正常模式
          }

          bOK = bOKReloc || bOKMM;  // 只要有一种方式追踪成功，则成功
        }
      }
    }

    // 更新参考关键帧
    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    /* 局部 BA */
    // If we have an initial estimation of the camera pose and matching. Track
    // the local map.
    // 如果运动模型或参考关键帧追踪成功，则进一步追踪局部地图，寻找 F 及其 MP 与
    // LocalMap 的匹配关系
    if (!mbOnlyTracking) {
      // SLAM 模式  追踪成功
      if (bOK) bOK = TrackLocalMap();
    } else {
      // 纯定位模式  追踪成功且追踪到足够 MP
      // mbVO true means that there are few matches to MapPoints in the map. We
      // cannot retrieve a local map and therefore we do not perform
      // TrackLocalMap(). Once the system relocalizes the camera we will use the
      // local map again.
      if (bOK && !mbVO) bOK = TrackLocalMap();
    }

    if (bOK)
      mState = OK;
    else
      mState = LOST;

    // Update drawer
    mpFrameDrawer->Update(this);

    /* 更新运动模型，关键帧判断 */
    // If tracking were good, check if we insert a keyframe
    if (bOK) {
      // 更新运动模型
      if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        mLastFrame.GetRotationInverse().copyTo(
            LastTwc.rowRange(0, 3).colRange(0, 3));  // Rwcl
        mLastFrame.GetCameraCenter().copyTo(
            LastTwc.rowRange(0, 3).col(3));  // twcl
        mVelocity = mCurrentFrame.mTcw *
                    LastTwc;  // Tcccl 两帧间的相对位姿态，并不是速度
      } else
        mVelocity = cv::Mat();  // 上一帧位姿缺失，速度失效，赋空矩阵

      mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

      // Clean VO matches
      // 清除 VO 匹配
      for (int i = 0; i < mCurrentFrame.N; i++) {
        // 遍历当前 F MP 观测，如果没有一个关键帧观测到它，就舍弃
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];  // 地图点与特征点对应
        if (pMP)
          // 如果 MP 没有被 KP 观测到，舍弃
          if (pMP->Observations() < 1) {
            mCurrentFrame.mvbOutlier[i] = false;  // 不是外点？
            mCurrentFrame.mvpMapPoints[i] =
                static_cast<MapPoint*>(NULL);  // 舍弃 MP 观测
          }
      }

      // 清空临时的地图点容器，释放内存
      for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(),
                                     lend = mlpTemporalPoints.end();
           lit != lend; lit++) {
        MapPoint* pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();

      /* 关键帧判断 */
      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrame()) CreateNewKeyFrame();

      // We allow points with high innovation (considererd outliers by the Huber
      // Function) pass to the new keyframe, so that bundle adjustment will
      // finally decide if they are outliers or not. We don't want next frame to
      // estimate its position with those points so we discard them in the
      // frame. 遍历当前 F MP
      // 观测，如果观测为外点，舍弃之，防止其在下一次追踪中被使用
      for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
      }
    }

    // 如果初始化后 5 个 KF 内追踪失败，重新进行初始化
    // Reset if the camera get lost soon after initialization
    if (mState == LOST) {
      if (mpMap->KeyFramesInMap() <= 5) {
        cout << "Track lost soon after initialisation, reseting..." << endl;
        mpSystem->Reset();
        return;
      }
    }

    // 如果当前 F 没有参考 KF
    if (!mCurrentFrame.mpReferenceKF)
      mCurrentFrame.mpReferenceKF = mpReferenceKF;

    // 更新上一帧，准备下次追踪
    mLastFrame = Frame(mCurrentFrame);
  }

  // Store frame pose information to retrieve the complete camera trajectory
  // afterwards. 保存帧位姿信息，用于恢复轨迹
  if (!mCurrentFrame.mTcw.empty()) {
    // 如果跟踪成功，保存相对于参考 KF 的位姿
    cv::Mat Tcr = mCurrentFrame.mTcw *
                  mCurrentFrame.mpReferenceKF->GetPoseInverse();  // Tcw * Twr
    mlRelativeFramePoses.push_back(Tcr);     // 相对于参考帧的位姿
    mlpReferences.push_back(mpReferenceKF);  // 参考关键帧指针
    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);  // 时间戳
    mlbLost.push_back(mState == LOST);
  } else {
    // 如果追踪失败，复制前一帧位姿信息
    // This can happen if tracking is lost
    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
    mlpReferences.push_back(mlpReferences.back());
    mlFrameTimes.push_back(mlFrameTimes.back());
    mlbLost.push_back(mState == LOST);
  }
}

/// @brief 双目、深度初始化
void Tracking::StereoInitialization() {
  // 如果提取特征点多于 500 个，进行初始化
  if (mCurrentFrame.N > 500) {
    // Set Frame pose to the origin
    // 当前 F 设置为原点
    mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

    // Create KeyFrame
    // 创建 KF
    KeyFrame* pKFini = new KeyFrame(
        mCurrentFrame, mpMap, mpKeyFrameDB);  // 从 mCurrentFrame 创建初始 KF

    // Insert KeyFrame in the map
    // 添加初始 KF 到地图
    mpMap->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    // 遍历初始 KF KP，对深度有效的 KP 创建 MP
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];  // 深度
      // 如果深度有效，计算其空间位置
      if (z > 0) {
        // 如果深度有效则添加，无效的点被舍弃
        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);  //
        // 新建 MP
        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
        // MP 与 KF 的相互关联
        pNewMP->AddObservation(pKFini, i);        // 为 MP 添加 KF 观测
        pKFini->AddMapPoint(pNewMP, i);           // 为 KF 添加 MP 观测
        pNewMP->ComputeDistinctiveDescriptors();  // 计算描述子
        pNewMP->UpdateNormalAndDepth();  // 更新平均观测方向和尺度无关距离
        mpMap->AddMapPoint(pNewMP);  // 添加 MP 到地图

        mCurrentFrame.mvpMapPoints[i] = pNewMP;  // 为当前 F 添加 MP 观测
      }
    }

    cout << "New map created with " << mpMap->MapPointsInMap() << " points"
         << endl;

    mpLocalMapper->InsertKeyFrame(pKFini);

    // 更新成员变量
    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFini;

    // 更新局部地图
    mvpLocalKeyFrames.push_back(pKFini);  // 将初始 KF 放入局部地图
    mvpLocalMapPoints =
        mpMap->GetAllMapPoints();  // 初始 KF 深度恢复的所有 MP 放入局部地图
    mpReferenceKF = pKFini;                // 参考 KF
    mCurrentFrame.mpReferenceKF = pKFini;  // 当前 F 参考 KF 为他自己

    // 更新地图信息
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

    mState = OK;
  }
}

/// @brief 释放初始化器
void Tracking::FreeInitializer() {
  delete mpInitializer;
  mpInitializer = nullptr;
}

/// @brief 单目初始化
void Tracking::MonocularInitialization() {
  /* 阶段一.创建初始化器与初始化参考帧 */
  // 如果未创建初始化器，则创建，并保留当前帧作为初始化参考帧，返回，等待下一帧
  // mpInitializer 的指针也作为初始化状态标志位使用
  if (!mpInitializer) {
    // 如果当前帧左图关键点数多于 100
    if (mCurrentFrame.mvKeys.size() > 100) {
      mInitialFrame = Frame(mCurrentFrame);
      mLastFrame = Frame(mCurrentFrame);
      mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

      mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);  // 创建初始化器
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);  // 清空初始化匹配
    }
    return;
  }

  /* 阶段二.多视图几何初始化 */
  // 输入图像左图关键点数小于 100，释放初始化器，返回，重新进行阶段一
  if ((int)mCurrentFrame.mvKeys.size() <= 100) {
    FreeInitializer();
    return;
  }

  ORBmatcher matcher(0.9, true);  // 匹配器
  /* 使用特征匹配器进行帧间特征点匹配 */
  int nmatches = matcher.SearchForInitialization(
      mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

  // 匹配少于 100 对，释放初始化器，返回阶段 1
  if (nmatches < 100) {
    FreeInitializer();
    return;
  }

  cv::Mat Rcw;                  // Current Camera Rotation
  cv::Mat tcw;                  // Current Camera Translation
  vector<bool> vbTriangulated;  // Triangulated Correspondences (mvIniMatches)

  /* 使用初始化器进行初始化、三角化 */
  if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D,
                                vbTriangulated)) {
    // 遍历匹配结果，如果 KP 没有成功三角化，则认为匹配失败
    for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
      if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
        mvIniMatches[i] = -1;
        nmatches--;
      }
    }

    // Set Frame Poses
    mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));  // 初始帧为原点
    cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
    Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
    tcw.copyTo(Tcw.rowRange(0, 3).col(3));
    mCurrentFrame.SetPose(Tcw);

    CreateInitialMapMonocular();  // 创建初始单目地图
  }
}

/// @brief 创建单目初始地图
void Tracking::CreateInitialMapMonocular() {
  // Create KeyFrames
  // 从初始帧和当前帧创建 KF
  KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
  KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  // 将 KF 加入地图
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  // 遍历 KP 匹配结果
  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    // 如果没有匹配结果，忽略 KP
    if (mvIniMatches[i] < 0) continue;

    // Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);  // MP 坐标 tw

    MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);  // 为 KP 创建 MP

    // 添加 KF 对 MP 观测
    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    // 添加 MP 的 KF 观测
    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    // 更新 MP 描述子、尺度无关距离、观测方向
    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

    // Add to Map
    // 将 MP 添加到地图
    mpMap->AddMapPoint(pMP);
  }

  // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  cout << "New Map created with " << mpMap->MapPointsInMap() << " points"
       << endl;

  /* 对初始地图进行全局 BA */
  // Optimizer::GlobalBundleAdjustemnt(mpMap, 20);
  CeresOptimizer::GlobalBundleAdjustment(mpMap, 20);

  // Set median depth to 1
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);  // 深度中位数
  float invMedianDepth = 1.0f / medianDepth;  // 深度中位数倒数

  // 如果深度中位数为负，或追踪到的有效地图点数小于 100，认为初始化错误，重置
  // Tracking
  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
    cout << "Wrong initialization, reseting..." << endl;
    Reset();
    return;
  }

  /* 尺度归一化，使 MP 深度中位数为数值 1 */
  // Scale initial baseline
  // 初始化基线缩放，即缩放当前帧和参考帧间的距离
  cv::Mat Tc2w = pKFcur->GetPose();
  Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
  pKFcur->SetPose(Tc2w);

  // Scale points
  // MP 坐标缩放
  vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    // 如果 MP 有效
    if (vpAllMapPoints[iMP]) {
      MapPoint* pMP = vpAllMapPoints[iMP];
      pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }
  }

  // 向 LocalMapping 添加 KF
  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);

  // 设置成员变量
  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKFcur;

  // 设置局部地图
  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

  mpMap->mvpKeyFrameOrigins.push_back(pKFini);

  mState = OK;
}

/// @brief 更新上一帧被替换的 MP
void Tracking::CheckReplacedInLastFrame() {
  // 遍历上一 F 观测 MP
  for (int i = 0; i < mLastFrame.N; i++) {
    MapPoint* pMP = mLastFrame.mvpMapPoints[i];

    if (pMP) {
      MapPoint* pRep = pMP->GetReplaced();
      // 如果地图点被替换，更新
      if (pRep) {
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}

/// @brief 根据参考关键帧估计相机位姿
/// @return 追踪结果
bool Tracking::TrackReferenceKeyFrame() {
  // Compute Bag of Words vector
  mCurrentFrame.ComputeBoW();

  // 下面的解释错误，没有创建 PnP 求解器，这里是直接调用优化器的
  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.7, true);
  vector<MapPoint*> vpMapPointMatches;

  // 词袋匹配函数，寻找当前 F KP 到参考 KF MP 的匹配
  int nmatches =
      matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

  if (nmatches < 15) return false;

  // 将词袋匹配到的参考 KF MP 作为当前帧地图点观测
  mCurrentFrame.mvpMapPoints = vpMapPointMatches;
  mCurrentFrame.SetPose(mLastFrame.mTcw);  // 初值使用上一帧

  // 这里有大问题!
  // Optimizer::PoseOptimization(&mCurrentFrame);
  CeresOptimizer::PoseOptimization(&mCurrentFrame);

  // Frame frame_test(mCurrentFrame);
  // auto ceres_res = CeresOptimizer::PoseOptimization(&frame_test);
  // if (abs(g2o_res - ceres_res) > 20) {
  //   std::cout << g2o_res << " " << ceres_res << std::endl;
  // }

  // Discard outliers
  // 外点剔除
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      // 此处外点标志位是在 Pose Optimization 优化中根据残差大小确定的
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        // 删除 F 对 MP 的观测
        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        // 清除 MP 成员变量
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      }
      // 如果 MP 有效
      else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  return nmatchesMap >= 10;
}

/// @brief 更新上一帧，纯定位模式下创建临时 MP
void Tracking::UpdateLastFrame() {
  // 为什么需要下面这个步骤，考虑到优化会改变结果？
  // Update pose according to reference keyframe
  KeyFrame* pRef = mLastFrame.mpReferenceKF;
  cv::Mat Tlr = mlRelativeFramePoses.back();

  mLastFrame.SetPose(Tlr * pRef->GetPose());  // 相对位姿到绝对位姿

  /* 纯定位模式下，为上一帧的近双目 KP 创建 MP */
  if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR ||
      !mbOnlyTracking)
    return;

  // Create "visual odometry" MapPoints
  // We sort points according to their measured depth by the stereo/RGB-D sensor
  // 为深度有效的近双目点 KP 创建 MP
  vector<pair<float, int> > vDepthIdx;  // 用于排序  v<p<depth, KP index>>
  vDepthIdx.reserve(mLastFrame.N);
  // 遍历所有 KP
  for (int i = 0; i < mLastFrame.N; i++) {
    float z = mLastFrame.mvDepth[i];
    if (z > 0) {
      vDepthIdx.push_back(make_pair(z, i));
    }
  }

  if (vDepthIdx.empty()) return;

  // 排序
  sort(vDepthIdx.begin(), vDepthIdx.end());

  // We insert all close points (depth<mThDepth)
  // If less than 100 close points, we insert the 100 closest ones.
  int nPoints = 0;
  for (size_t j = 0; j < vDepthIdx.size(); j++) {
    int i = vDepthIdx[j].second;

    bool bCreateNew = false;

    // 如果这个 KP 已经对应到有效的 MP，忽略   调用这个函数时貌似还没有对 F
    // 进行过匹配
    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
    if (!pMP)
      bCreateNew = true;
    else if (pMP->Observations() < 1) {
      bCreateNew = true;
    }

    if (bCreateNew) {
      cv::Mat x3D = mLastFrame.UnprojectStereo(i);
      // 为近双目点创建临时地图点，注意这些地图点没有添加到 Map 中，且 KF
      // 观测数为 0
      MapPoint* pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

      mLastFrame.mvpMapPoints[i] = pNewMP;

      mlpTemporalPoints.push_back(pNewMP);
      nPoints++;
    } else {
      nPoints++;
    }

    if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
  }
}

/// @brief 根据匀速模型估计相机位姿
/// @return
bool Tracking::TrackWithMotionModel() {
  ORBmatcher matcher(0.9, true);

  // 更新上一帧，纯定位模式下创建临时 MP
  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  UpdateLastFrame();

  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  // 清除当前帧地图点观测
  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint*>(NULL));

  // Project points seen in previous frame
  int th;
  if (mSensor != System::STEREO)
    th = 15;
  else
    th = 7;

  /* 帧间匹配 */
  int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th,
                                            mSensor == System::MONOCULAR);

  // If few matches, uses a wider window search
  if (nmatches < 20) {
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));
    nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th,
                                          mSensor == System::MONOCULAR);
  }

  if (nmatches < 20) return false;

  /* 优化 */
  // Optimize frame pose with all matches
  // Frame frame_test(mCurrentFrame);
  // auto g2o_res = Optimizer::PoseOptimization(&frame_test);
  CeresOptimizer::PoseOptimization(&mCurrentFrame);
  // std::cout << "g2o: " << g2o_res << " ceres: " << ceres_res << std::endl;
  // std::cout << std::endl;

  // Discard outliers
  int nmatchesMap = 0;  // 地图点匹配
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      }
      // 坏点或临时点
      else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  // 纯定位模式下，如果追踪到地图点数量不足，但是帧见匹配
  if (mbOnlyTracking) {
    mbVO = nmatchesMap < 10;
    return nmatches > 20;
  }

  return nmatchesMap >= 10;
}

/// @brief 追踪局部地图
/// @return
bool Tracking::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the
  // frame. We retrieve the local map and try to find matches to points in the
  // local map.

  // 更新局部地图
  UpdateLocalMap();

  // 搜索局部点
  SearchLocalPoints();

  // Optimize Pose
  // auto g2o_res = Optimizer::PoseOptimization(&mCurrentFrame);
  CeresOptimizer::PoseOptimization(&mCurrentFrame);

  // Frame frame_test(mCurrentFrame);
  // auto ceres_res = CeresOptimizer::PoseOptimization(&frame_test);
  // if (abs(g2o_res - ceres_res) > 20) {
  //   std::cout << "Track local map res: " << g2o_res << " " << ceres_res << std::endl;
  // }

    // break here

  mnMatchesInliers = 0;  // 匹配内点数量

  // Update MapPoints Statistics
  // 遍历当前 F KP
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (!mCurrentFrame.mvbOutlier[i]) {
        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            mnMatchesInliers++;
        } else
          mnMatchesInliers++;
      } else if (mSensor == System::STEREO)
        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    }
  }

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  // 重定位 1s 内的检测更加严格
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
      mnMatchesInliers < 50)
    return false;

  if (mnMatchesInliers < 30)
    return false;
  else
    return true;
}

/// @brief 关键帧判断
/// @return
bool Tracking::NeedNewKeyFrame() {
  // 纯定位模式下不需要创建 KF
  if (mbOnlyTracking) return false;

  // 当 LocalMapping 被 LoopClosing 中断时，不创建 KF
  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    return false;

  const int nKFs = mpMap->KeyFramesInMap();  // KF 总数

  // 如果距离上次 KF 创建没有走过足够 F，不创建 KF
  // Do not insert keyframes if not enough frames have passed from last
  // relocalisation
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
    return false;

  // Tracked MapPoints in the reference keyframe
  int nMinObs = 3;
  if (nKFs <= 2) nMinObs = 2;
  int nRefMatches =
      mpReferenceKF->TrackedMapPoints(nMinObs);  // 参考 KF 有效 MP 追踪数量

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle =
      mpLocalMapper->AcceptKeyFrames();  // LocalMapping 是否接受 KF

  // Check how many "close" points are being tracked and how many could be
  // potentially created.
  int nNonTrackedClose = 0;  // 未匹配的近双目点
  int nTrackedClose = 0;     // 已匹配的近双目点
  // 双目或深度
  if (mSensor != System::MONOCULAR) {
    // 遍历 KF
    for (int i = 0; i < mCurrentFrame.N; i++) {
      // 如果深度有效，且为近双目点
      if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
        // 对于存在 MP 匹配，且匹配不为外点的 KP，记为已匹配
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
          nTrackedClose++;
        else
          nNonTrackedClose++;
      }
    }
  }

  // 当已匹配的近双目点过少且未匹配的近双目点过多时，设置需要添加近双目点标志位
  bool bNeedToInsertClose = (nTrackedClose < 100) &&
                            (nNonTrackedClose > 70);  // 需要添加近双目点标志位

  // Thresholds
  float thRefRatio = 0.75f;  // 参考 KF 阈值
  if (nKFs < 2) thRefRatio = 0.4f;

  if (mSensor == System::MONOCULAR) thRefRatio = 0.9f;

  // 条件1a  距离上次创建 KF 已超过最大创建间隔
  // Condition 1a: More than "MaxFrames" have passed from last keyframe
  // insertion
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
  // 条件1b  距离上次创建 KF 已超过最小创建间隔，且 LocalMapping 接收 KF
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames &&
                    bLocalMappingIdle);
  // 条件1c  非单目下，匹配到 MP 过少或需要添加近双目点
  // Condition 1c: tracking is weak
  const bool c1c =
      mSensor != System::MONOCULAR &&
      (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
  // 条件2  匹配到 MP 过少或需要添加近双目点，相比下 VO 匹配点更多
  // Condition 2: Few tracked points compared to reference keyframe. Lots of
  // visual odometry compared to map matches.
  const bool c2 =
      ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
       mnMatchesInliers > 15);

  if ((c1a || c1b || c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle) {
      return true;
    } else {
      // 如果 LocalMapping 不接收 KF，则发送信号中断 BA
      mpLocalMapper->InterruptBA();
      if (mSensor != System::MONOCULAR) {
        if (mpLocalMapper->KeyframesInQueue() < 3)
          return true;
        else
          return false;
      } else
        return false;
    }
  } else
    return false;
}

/// @brief 创建新关键帧
void Tracking::CreateNewKeyFrame() {
  // 设置 LocalMapping 为拒绝中断
  if (!mpLocalMapper->SetNotStop(true)) return;

  // 创建 KF
  KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  // 参考 KF 设置为新 KF
  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  // 非单目模式下，为近双目点创建 MP
  // 如果近双目点数量少于 100，则额外为最近的双目点创建 MP，使观测到的 MP 总数为
  // 100
  if (mSensor != System::MONOCULAR) {
    mCurrentFrame.UpdatePoseMatrices();  // 更新位姿信息

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    vector<pair<float, int> > vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        vDepthIdx.push_back(make_pair(z, i));
      }
    }

    if (!vDepthIdx.empty()) {
      // 按照深度排序
      sort(vDepthIdx.begin(), vDepthIdx.end());

      int nPoints = 0;  // MP 总数
      for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        // 如果 KP 存在有效的 MP 观测，忽略
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP)
          bCreateNew = true;
        else if (pMP->Observations() < 1) {
          bCreateNew = true;
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }

        // 如果需要创建 MP
        if (bCreateNew) {
          cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);    // twp
          MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);  // new MP
          // 更新观测关系，添加 MP 到地图
          pNewMP->AddObservation(pKF, i);
          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          mpMap->AddMapPoint(pNewMP);

          mCurrentFrame.mvpMapPoints[i] = pNewMP;
          nPoints++;
        } else {
          nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
      }
    }
  }

  // 添加 KF
  mpLocalMapper->InsertKeyFrame(pKF);

  // 恢复 LocalMapping
  mpLocalMapper->SetNotStop(false);

  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKF;
}

/// @brief 匹配当前 F 与局部 MP
void Tracking::SearchLocalPoints() {
  /* 检查局部 MP 是否在当前 F 视角截锥体内 */
  // 忽略已经匹配的 MP
  // Do not search map points already matched
  for (vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(),
                                   vend = mCurrentFrame.mvpMapPoints.end();
       vit != vend; vit++) {
    MapPoint* pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        *vit = static_cast<MapPoint*>(NULL);
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        pMP->mbTrackInView = false;
      }
    }
  }

  int nToMatch = 0;

  // 检查 MP 可视性，即是否位于当前 F 视角截椎体内
  // Project points in frame and check its visibility
  for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(),
                                   vend = mvpLocalMapPoints.end();
       vit != vend; vit++) {
    MapPoint* pMP = *vit;
    if (pMP->mnLastFrameSeen == mCurrentFrame.mnId) continue;
    if (pMP->isBad()) continue;
    // Project (this fills MapPoint variables for matching)
    if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
      pMP->IncreaseVisible();
      nToMatch++;
    }
  }

  if (nToMatch > 0) {
    ORBmatcher matcher(0.8);
    int th = 1;
    if (mSensor == System::RGBD) th = 3;
    // If the camera has been relocalised recently, perform a coarser search
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) th = 5;
    // 投影匹配
    matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
  }
}

/// @brief 更新局部地图
void Tracking::UpdateLocalMap() {
  // This is for visualization
  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

/// @brief 更新局部点
void Tracking::UpdateLocalPoints() {
  /* 将局部 KF 的 MP 观测作为局部 MP */
  // 清空局部 MP
  mvpLocalMapPoints.clear();

  // 遍历局部 KF
  for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                         itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    KeyFrame* pKF = *itKF;
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    // 遍历局部 KF MP
    for (vector<MapPoint*>::const_iterator itMP = vpMPs.begin(),
                                           itEndMP = vpMPs.end();
         itMP != itEndMP; itMP++) {
      MapPoint* pMP = *itMP;
      if (!pMP) continue;
      if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) continue;
      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      }
    }
  }
}

/// @brief 更新局部关键帧
void Tracking::UpdateLocalKeyFrames() {
  /* 从当前 F MP 寻找共视 KF，加入局部 KF */
  // Each map point vote for the keyframes in which it has been observed
  map<KeyFrame*, int> keyframeCounter;  // KF 计数器
  // 遍历当前 F MP 观测
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
      if (!pMP->isBad()) {
        const map<KeyFrame*, size_t> observations =
            pMP->GetObservations();  // 观测到 MP 的 KF  map<KF ptr, KP idx>
        for (map<KeyFrame*, size_t>::const_iterator it = observations.begin(),
                                                    itend = observations.end();
             it != itend; it++)
          keyframeCounter[it->first]++;
      } else {
        mCurrentFrame.mvpMapPoints[i] = NULL;
      }
    }
  }

  if (keyframeCounter.empty()) return;

  // 统计最大共视数的 KF 作为参考 KF 使用
  int max = 0;                                      // 最大共视数
  KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);  // 最大共视数 KF

  // 清空局部地图，预分配内存
  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map. Also
  // check which keyframe shares most points 遍历 KF 计数器，将共视 KF
  // 加入局部地图
  for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(),
                                           itEnd = keyframeCounter.end();
       it != itEnd; it++) {
    KeyFrame* pKF = it->first;

    if (pKF->isBad()) continue;

    if (it->second > max) {
      max = it->second;
      pKFmax = pKF;
    }

    mvpLocalKeyFrames.push_back(it->first);
    pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }

  // 下面的循环在迭代过程中动态增加 vector
  // 元素，会出问题。但是提前分配了内存，也可能不会出问题，总之很奇怪。最好分两个
  // vector 写吧 Include also some not-already-included keyframes that are
  // neighbors to already-included keyframes 将共视 KF 的共视 KF、生成树相邻 KF
  // 添加到局部地图中
  for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                         itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    // Limit the number of keyframes
    if (mvpLocalKeyFrames.size() > 80) break;

    KeyFrame* pKF = *itKF;

    // 添加最佳共视 KF 到局部地图
    const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

    for (vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(),
                                           itEndNeighKF = vNeighs.end();
         itNeighKF != itEndNeighKF; itNeighKF++) {
      KeyFrame* pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad()) {
        if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pNeighKF);
          pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    // 添加生成树子 KF 到局部地图
    const set<KeyFrame*> spChilds = pKF->GetChilds();
    for (set<KeyFrame*>::const_iterator sit = spChilds.begin(),
                                        send = spChilds.end();
         sit != send; sit++) {
      KeyFrame* pChildKF = *sit;
      if (!pChildKF->isBad()) {
        if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pChildKF);
          pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    // 添加父 KF 到局部地图
    KeyFrame* pParent = pKF->GetParent();
    if (pParent) {
      if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pParent);
        pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }
  }

  // 如果存在最佳共视 KF，将其作为参考 KF
  if (pKFmax) {
    mpReferenceKF = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}

/// @brief 重定位
/// @return
bool Tracking::Relocalization() {
  // Compute Bag of Words Vector
  mCurrentFrame.ComputeBoW();

  /* 1.获取重定位候选 KF */

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for
  // relocalisation
  vector<KeyFrame*> vpCandidateKFs =
      mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

  if (vpCandidateKFs.empty()) return false;

  const int nKFs = vpCandidateKFs.size();  // 候选 KF 数量

  /* 2.初次匹配，词袋匹配 */
  // We perform first an ORB matching with each candidate
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.75, true);

  vector<PnPsolver*> vpPnPsolvers;  // 求解器
  vpPnPsolvers.resize(nKFs);

  vector<vector<MapPoint*> > vvpMapPointMatches;  // 当前 F 到候选 KF MP 匹配
  vvpMapPointMatches.resize(nKFs);

  vector<bool> vbDiscarded;  // 抛弃候选 KF 标志位
  vbDiscarded.resize(nKFs);

  int nCandidates = 0;  // 第二轮候选 KF 数量

  // 遍历全部候选 KF
  for (int i = 0; i < nKFs; i++) {
    KeyFrame* pKF = vpCandidateKFs[i];
    if (pKF->isBad())
      vbDiscarded[i] = true;
    else {
      // 词袋匹配
      int nmatches =
          matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
      // 如果匹配点数量足够，创建 PnP 求解器
      if (nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        PnPsolver* pSolver =
            new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
        pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
        vpPnPsolvers[i] = pSolver;
        nCandidates++;
      }
    }
  }

  /* 3.再次匹配，先利用 PnP 求解初值，再进行投影匹配 */
  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool bMatch = false;
  ORBmatcher matcher2(0.9, true);

  // 循环，直到寻找到匹配或候选 KF 全部淘汰
  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nKFs; i++) {
      if (vbDiscarded[i]) continue;

      // Perform 5 Ransac Iterations
      vector<bool> vbInliers;  // 内点标志位
      int nInliers;            // 内点总数
      bool bNoMore;

      PnPsolver* pSolver = vpPnPsolvers[i];
      // 进行 5 次 RANSAC 迭代
      cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

      // If Ransac reachs max. iterations discard keyframe
      // 如果 RANSAC 达到最大迭代次数，或本轮迭代未得到有效结果，淘汰候选 KF
      if (bNoMore || Tcw.empty()) {
        vbDiscarded[i] = true;
        nCandidates--;
        // 源码没有下面这行。PnPsolver 求解失败时不应该继续计算，否则报段错误
        continue;
      }

      // 记录位姿计算结果，作为优化初值使用
      Tcw.copyTo(mCurrentFrame.mTcw);

      set<MapPoint*> sFound;  // 当前 F 观测到的 MP

      const int np = vbInliers.size();

      // 添加当前 F 对 MP 的观测，在优化中使用
      // 遍历当前 F 所有 KP
      for (int j = 0; j < np; j++) {
        // 如果 KP 为对应内点
        if (vbInliers[j]) {
          mCurrentFrame.mvpMapPoints[j] =
              vvpMapPointMatches[i][j];  // 添加 MP 观测
          sFound.insert(vvpMapPointMatches[i][j]);
        } else
          mCurrentFrame.mvpMapPoints[j] = NULL;
      }

      // Motion-only BA
      // int nGood = Optimizer::PoseOptimization(&mCurrentFrame);  // 内点数量
      int nGood = CeresOptimizer::PoseOptimization(&mCurrentFrame);

      // Frame frame_test(mCurrentFrame);
      // auto ceres_res = CeresOptimizer::PoseOptimization(&frame_test);
      // if (abs(nGood - ceres_res) > 20) {
      //   std::cout << nGood << " " << ceres_res << std::endl;
      // }

      if (nGood < 10) continue;

      // 遍历 KP，清除外点观测
      for (int io = 0; io < mCurrentFrame.N; io++)
        if (mCurrentFrame.mvbOutlier[io])
          mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

      // 如果内点很少，进行投影匹配，重新优化
      // If few inliers, search by projection in a coarse window and optimize
      // again
      if (nGood < 50) {
        // 重定位投影匹配
        int nadditional = matcher2.SearchByProjection(
            mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

        if (nadditional + nGood >= 50) {
          // Motion-only BA
          // nGood = Optimizer::PoseOptimization(&mCurrentFrame);
          nGood = CeresOptimizer::PoseOptimization(&mCurrentFrame);

          // Frame frame_test(mCurrentFrame);
          // auto ceres_res = CeresOptimizer::PoseOptimization(&frame_test);
          // if (abs(nGood - ceres_res) > 20) {
          //   std::cout << nGood << " " << ceres_res << std::endl;
          // }

          // If many inliers but still not enough, search by projection again in
          // a narrower window the camera has been already optimized with many
          // points 如果内点还是不够，在小窗口中再次匹配、优化
          if (nGood > 30 && nGood < 50) {
            sFound.clear();
            for (int ip = 0; ip < mCurrentFrame.N; ip++)
              if (mCurrentFrame.mvpMapPoints[ip])
                sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
            nadditional = matcher2.SearchByProjection(
                mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

            // Final optimization
            if (nGood + nadditional >= 50) {
              // nGood = Optimizer::PoseOptimization(&mCurrentFrame);
              nGood = CeresOptimizer::PoseOptimization(&mCurrentFrame);

              // Frame frame_test(mCurrentFrame);
              // auto ceres_res = CeresOptimizer::PoseOptimization(&frame_test);
              // if (abs(nGood - ceres_res) > 20) {
              //   std::cout << nGood << " " << ceres_res << std::endl;
              // }

              for (int io = 0; io < mCurrentFrame.N; io++)
                if (mCurrentFrame.mvbOutlier[io])
                  mCurrentFrame.mvpMapPoints[io] = NULL;
            }
          }
        }
      }

      // If the pose is supported by enough inliers stop ransacs and continue
      // 只要有一个候选 KF 与当前 F 有足够多匹配点，结束 RANSAC 采样
      if (nGood >= 50) {
        bMatch = true;
        break;
      }
    }
  }

  if (!bMatch) {
    return false;
  } else {
    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
  }
}

/// @brief 重置
void Tracking::Reset() {
  cout << "System Reseting" << endl;
  if (mpViewer) {
    mpViewer->RequestStop();
    while (!mpViewer->isStopped()) usleep(3000);
  }

  // Reset Local Mapping
  cout << "Reseting Local Mapper...";
  mpLocalMapper->RequestReset();
  cout << " done" << endl;

  // Reset Loop Closing
  cout << "Reseting Loop Closing...";
  mpLoopClosing->RequestReset();
  cout << " done" << endl;

  // Clear BoW Database
  cout << "Reseting Database...";
  mpKeyFrameDB->clear();
  cout << " done" << endl;

  // Clear Map (this erase MapPoints and KeyFrames)
  mpMap->clear();

  KeyFrame::nNextId = 0;
  Frame::nNextId = 0;
  mState = NO_IMAGES_YET;

  if (mpInitializer) {
    delete mpInitializer;
    mpInitializer = static_cast<Initializer*>(NULL);
  }

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();

  if (mpViewer) mpViewer->Release();
}

/// @brief 加载新配置
/// @param strSettingPath
void Tracking::ChangeCalibration(const string& strSettingPath) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  mbf = fSettings["Camera.bf"];

  Frame::mbInitialComputations = true;
}

/// @brief 设置纯定位模式
/// @param flag
void Tracking::InformOnlyTracking(const bool& flag) { mbOnlyTracking = flag; }

}  // namespace ORB_SLAM2
