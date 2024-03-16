#include <chrono>
#include <condition_variable>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <thread>

#include "System.h"

using namespace std::chrono_literals;

void CheckSensorOptions(rs2::sensor &sensor);

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: ./d435i_rgbd path_to_vocabulary path_to_settings" << endl;
    return 1;
  }
  /* 查询设备 */
  rs2::context ctx;
  rs2::device_list devices = ctx.query_devices();  // 从上下文获取传感器
  std::cout << devices.size() << " devices in total." << std::endl;

  /* 查询传感器 */
  rs2::device selected_device = devices[0];
  std::vector<rs2::sensor> sensors = selected_device.query_sensors();
  std::cout << sensors.size() << " sensors in total. They are:" << std::endl;

  for (size_t i = 0; i < sensors.size(); ++i) {
    rs2::sensor sensor = sensors[i];
    std::cout << "  " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
    // CheckSensorOptions(sensor);
  }

  /* 传感器参数设置 */
  rs2::depth_sensor depth_s = selected_device.first<rs2::depth_sensor>();
  rs2::color_sensor color_s = selected_device.first<rs2::color_sensor>();
  rs2::motion_sensor motion_s = selected_device.first<rs2::motion_sensor>();
  depth_s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
  depth_s.set_option(RS2_OPTION_EMITTER_ENABLED, 1);
  color_s.set_option(RS2_OPTION_EXPOSURE, 300.0f);
  motion_s.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION, 0);

  /* 管道配置 */
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
  cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
  cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

  /* 管道回调函数 */
  std::mutex mutex_global;
  bool frame_ready = false;
  std::condition_variable condV;
  rs2::frameset frame_global;

  auto frame_callback = [&](const rs2::frame &frame) {
    std::unique_lock<std::mutex> lock(mutex_global);

    if (rs2::frameset frame_temp = frame.as<rs2::frameset>()) {
      frame_global = frame_temp;
      frame_ready = true;
      lock.unlock();
      condV.notify_all();
    }
  };

  /* 开启管道，获取简述 */
  rs2::pipeline pipe;

  rs2::pipeline_profile profile = pipe.start(cfg, frame_callback);

  rs2::stream_profile color_stream = profile.get_stream(RS2_STREAM_COLOR);
  rs2_intrinsics color_instrinsics =
      color_stream.as<rs2::video_stream_profile>().get_intrinsics();

  int image_height = color_instrinsics.height;
  int image_width = color_instrinsics.width;
  std::cout << "Instrinsic parameters: " << std::endl;
  std::cout << "  height: " << color_instrinsics.height << std::endl;
  std::cout << "  width: " << color_instrinsics.width << std::endl;
  std::cout << "  fx: " << color_instrinsics.fx << std::endl;
  std::cout << "  fy: " << color_instrinsics.fy << std::endl;
  std::cout << "  model: " << color_instrinsics.model << std::endl;

  rs2::colorizer color_map;
  rs2::align aling_tool(RS2_STREAM_COLOR);

  ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);

  while (true) {
    rs2::frameset frame_local;
    {
      std::unique_lock<std::mutex> lock(mutex_global);
      if (!frame_ready) condV.wait(lock);
      frame_local = frame_global;
      frame_ready = false;
    }

    rs2::frameset frame_alinged = aling_tool.process(frame_local);

    rs2::video_frame color_frame = frame_alinged.get_color_frame();
    rs2::depth_frame depth_frame = frame_alinged.get_depth_frame();

    cv::Mat color_img(cv::Size(image_width, image_height), CV_8UC3,
                      (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::Mat depth_img(cv::Size(image_width, image_height), CV_16U,
                      (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

    if (color_img.data && depth_img.data){

      SLAM.TrackRGBD(color_img, depth_img, frame_local.get_timestamp() * 1e-3);
      // SLAM.TrackMonocular(color_img, frame_local.get_timestamp() * 1e-3);
    }


  }

  pipe.stop();
  SLAM.Shutdown();

  return 0;
}

void CheckSensorOptions(rs2::sensor &sensor) {
  std::cout << "  Available options:" << std::endl;
  for (size_t j = 0; j < RS2_OPTION_COUNT; ++j) {
    rs2_option option_type = static_cast<rs2_option>(j);
    // std::cout << "Option type: " << option_type << std::endl;
    if (sensor.supports(option_type)) {
      std::cout << "    " << sensor.get_option_name(option_type) << ": "
                << sensor.get_option_description(option_type) << std::endl;
      std::cout << "    Current value: " << sensor.get_option(option_type)
                << std::endl;
    }
  }
}
