// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifdef SVO_USE_ROS
#include <vikit/params_helper.h>
#endif
#include <svo/config.h>
#include <opencv2/opencv.hpp>

namespace svo {

Config::Config() :
#ifdef SVO_USE_ROS
    trace_name(vk::getParam<string>("svo/trace_name", "svo")),
    trace_dir(vk::getParam<string>("svo/trace_dir", "/tmp")),
    n_pyr_levels(vk::getParam<int>("svo/n_pyr_levels", 3)),
    core_n_kfs(vk::getParam<int>("svo/core_n_kfs", 3)),
    map_scale(vk::getParam<double>("svo/map_scale", 1.0)),
    grid_size(vk::getParam<int>("svo/grid_size", 30)),
    init_min_disparity(vk::getParam<double>("svo/init_min_disparity", 50.0)),
    init_min_tracked(vk::getParam<int>("svo/init_min_tracked", 50)),
    init_min_inliers(vk::getParam<int>("svo/init_min_inliers", 40)),
    klt_max_level(vk::getParam<int>("svo/klt_max_level", 4)),
    klt_min_level(vk::getParam<int>("svo/klt_min_level", 2)),
    reproj_thresh(vk::getParam<double>("svo/reproj_thresh", 2.0)),
    poseoptim_thresh(vk::getParam<double>("svo/poseoptim_thresh", 2.0)),
    poseoptim_num_iter(vk::getParam<int>("svo/poseoptim_num_iter", 10)),
    structureoptim_max_pts(vk::getParam<int>("svo/structureoptim_max_pts", 20)),
    structureoptim_num_iter(vk::getParam<int>("svo/structureoptim_num_iter", 5)),
    loba_thresh(vk::getParam<double>("svo/loba_thresh", 2.0)),
    loba_robust_huber_width(vk::getParam<double>("svo/loba_robust_huber_width", 1.0)),
    loba_num_iter(vk::getParam<int>("svo/loba_num_iter", 0)),
    winba_num_iter(vk::getParam<int>("svo/winba_num_iter", 0)),
    kfselect_mindist(vk::getParam<double>("svo/kfselect_mindist", 0.12)),
    triang_min_corner_score(vk::getParam<double>("svo/triang_min_corner_score", 20.0)),
    triang_half_patch_size(vk::getParam<int>("svo/triang_half_patch_size", 4)),
    subpix_n_iter(vk::getParam<int>("svo/subpix_n_iter", 10)),
    max_n_kfs(vk::getParam<int>("svo/max_n_kfs", 10)),
    img_imu_delay(vk::getParam<double>("svo/img_imu_delay", 0.0)),
    max_fts(vk::getParam<int>("svo/max_fts", 120)),
    quality_min_fts(vk::getParam<int>("svo/quality_min_fts", 50)),
    quality_max_drop_fts(vk::getParam<int>("svo/quality_max_drop_fts", 40))
#else
    trace_name("svo"),
    trace_dir("/tmp"),
    n_pyr_levels(3), //3
    core_n_kfs(3),
    map_scale(1.0),
    grid_size(25),                    // 75
    init_min_disparity(50.0),
    init_min_tracked(50),
    init_min_inliers(40),
    klt_max_level(4), // 4      //6
    klt_min_level(2),  // 2      //4
    reproj_thresh(2.0),  // 2  //6.0
    poseoptim_thresh(2.0),
    poseoptim_num_iter(10),
    structureoptim_max_pts(20),
    structureoptim_num_iter(5),
    loba_thresh(2.0),
    loba_robust_huber_width(1.0),
    loba_num_iter(0),
    winba_num_iter(0),
    kfselect_mindist(0.12),   // 0.12
    triang_min_corner_score(5.0),   // 20
    triang_half_patch_size(4),
    subpix_n_iter(10),
    max_n_kfs(30),   // 10
    img_imu_delay(0.0),
    max_fts(180),    //120
    quality_min_fts(30),  //50   //20
    quality_max_drop_fts(40)
#endif
    {   ///调参的时候需要注意，参数改变的路经:
        cv::FileStorage fsConfig("/home/gzh/SVIO_rebuild/svio_1.1/test/config_param.yaml", cv::FileStorage::READ);
        if (!fsConfig.isOpened()) {
            return;
        }
        int data_temp;
        fsConfig["start_Frame"] >> start_Frame;                       std::cout << "start_Frame=" << start_Frame << std::endl;
        fsConfig["vio_init_stoptime"] >> vio_init_stoptime;           std::cout << "vio_init_stoptime=" << vio_init_stoptime << std::endl;
        fsConfig["vio_init_scale"] >> vio_init_scale;                 std::cout << "vio_init_scale=" << vio_init_scale << std::endl;
        fsConfig["kf_reproj_cnt"] >> kf_reproj_cnt;                   std::cout << "kf_reproj_cnt=" << kf_reproj_cnt << std::endl;
        fsConfig["kf_max_pixeldist"] >> kf_max_pixeldist;             std::cout << "kf_max_pixeldist" << kf_max_pixeldist << std::endl;
        fsConfig["use_imu_prior"] >> use_imu_prior;                   std::cout <<"use_imu_prior="<< use_imu_prior<<std::endl;
        fsConfig["fixedkf_obs_minftr_num"] >>fixedkf_obs_minftr_num;  std::cout<<"fixedkf_obs_minftr_num="<<fixedkf_obs_minftr_num<<std::endl;
        fsConfig["Edge_weight_PVRPoint"] >> Edge_weight_PVRPoint;     std::cout<<"Edge_weight_PVRPoint="<<Edge_weight_PVRPoint<<std::endl;
        fsConfig["seed_max_kf_n"] >> seed_max_kf_n;                   std::cout<<"seed_max_kf_n="<<seed_max_kf_n<<std::endl;


        fsConfig["max_n_kfs"] >> data_temp;       max_n_kfs = data_temp;       std::cout << "max_n_kfs=" << max_n_kfs << std::endl;
        fsConfig["loba_num_iter"] >> data_temp;   loba_num_iter = data_temp;   std::cout << "loba_num_iter=" << loba_num_iter << std::endl;
        fsConfig["winba_num_iter"] >>data_temp;   winba_num_iter = data_temp;  std::cout<<"winba_num_iter="<<winba_num_iter<<std::endl;
        fsConfig["max_fts"] >> data_temp;         max_fts = data_temp;         std::cout << "max_fts=" << max_fts << std::endl;
        fsConfig["quality_min_fts"] >> data_temp; quality_min_fts = data_temp; std::cout << "quality_min_fts=" << quality_min_fts << std::endl;
        fsConfig["core_n_kfs"] >>data_temp;       core_n_kfs = data_temp;      std::cout<<"core_n_kfs="<<core_n_kfs<<std::endl;

    }
Config& Config::getInstance()
{
  static Config instance; // Instantiated on first use and guaranteed to be destroyed
  return instance;
}

} // namespace svo

