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

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <svo/math_lib.h>
#include <svo/camera_model.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>
#include <fstream>
#include <svo/slamviewer.h>
#include <thread>

#include "svo/Data_EuRoc.h"
#include "../src/IMU/imudata.h"
#include <svo/vio_init.h>

using namespace cv;
namespace svo {

    void DistortImg( cv::Mat &img, bool flag){

        //  通过cv::FileStorage fsSetting()传入配置文件路径的读取 校正参数 (Read rectification parameters)
        //cv::FileStorage fsSettings("/home/gzh/datasets/EuRoc/mav0/cam0/EuRoC.yaml", cv::FileStorage::READ);
        cv::FileStorage fsSettings("/home/gzh/datasets/EuRoc/stereo_EuRoC.yaml",cv::FileStorage::READ);
        if (!fsSettings.isOpened()) {
            //cerr << "ERROR: Wrong path to settings" << endl;
            return;
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() ||
            D_r.empty() || rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return;
        }

        cv::Mat M1l, M2l, M1r, M2r;
        if(flag == 0){
            //使用左相机图像　－－校正
            cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F,
                                        M1l, M2l);
            // input,  output
            cv::remap(img, img, M1l, M2l, cv::INTER_LINEAR);//使用矫正后的图像
        }else{
            //使用左相机图像　－－校正
            cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F,
                                        M1r, M2r);
            // input,  output
            cv::remap(img, img, M1r, M2r, cv::INTER_LINEAR);//使用矫正后的图像
        }

    }

    double GetRealTime(const string str){  //timestamp(string)-->(double)

        return stod(str)/1e9;
    }

    void OutPutFile( svo::FrameHandlerMono *vo_, EuRocData* dataset, int img_id ){

#ifdef USE_IMU
        if (!vo_->vioInitFinish)return;
#endif
        // show tracking quality
        if (vo_->lastFrame() != NULL) {
            std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                      << "#Features: " << vo_->lastNumObservations() << " \n";

            // access the pose of the camera via vo_->lastFrame()->T_f_w_.
//            std::cout << "Frame pose: " << vo_->lastFrame()->T_f_w_ << std::endl;

            Eigen::Vector3d Pos_W;//position in world frame
            Pos_W = vo_->lastFrame()->T_f_w_.inverse().translation();
            Eigen::Quaterniond q = Eigen::Quaterniond(vo_->lastFrame()->T_f_w_.rotation_matrix());

            //输出数据保存路经
            ofstream outfile("/home/gzh/compare_test/data_test_pipel_euroc.txt", ios::app);
            if (outfile.fail()) {
                cout << " putout file svo_0.1 FAIL!!!" << endl;
            } else {
                outfile << setprecision(19) << setw(19) << vo_->lastFrame()->timestamp_ << " " <<
                        Pos_W[0] << " " <<
                        Pos_W[1] << " " <<
                        Pos_W[2] << " " <<

                        q.coeffs().transpose().x() << " " <<
                        q.coeffs().transpose().y() << " " <<
                        q.coeffs().transpose().z() << " " <<
                        q.coeffs().transpose().w() << endl;
                outfile.close();
            }
        }
    }

    class BenchmarkNode {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EuRocData *dataset;
//        svo::AbstractCamera *cam_;
//        svo::AbstractCamera *cam_r_;
//        svo::PinholeCamera *cam_pinhole_;  //　PinholeCamera：针孔相机模型　继承了AbstractCamera
        svo::PinholeCamera *cam_;            //gzh change
        svo::FrameHandlerMono *vo_;

        SLAM_VIEWER::Viewer *viewer_;
        std::thread *viewer_thread_;

        svo::VioInitialization *vioinit_;
        std::thread *vioinit_thread_;

    public:
        BenchmarkNode();

        ~BenchmarkNode();

        void runFromFolder();
    };

    BenchmarkNode::BenchmarkNode() {

        //dataset = new EuRocData("/home/gzh/datasets/EuRoc/mav0");
        //dataset = new EuRocData("/home/gzh/datasets/EuRoc/MH_02_easy/mav0");
        dataset = new EuRocData("/home/gzh/datasets/EuRoc/MH_03_medium/mav0", 2, 1);
        //dataset = new EuRocData("/home/gzh/datasets/EuRoc/MH_04_difficult/mav0");
        //dataset = new EuRocData("/home/gzh/datasets/EuRoc/V1_01_easy/mav0");
        //dataset = new EuRocData("/home/gzh/datasets/EuRoc/V1_02_medium/mav0");
        //dataset = new EuRocData("/home/gzh/datasets/EuRoc/V2_01_easy/mav0");


        cam_ = new svo::PinholeCamera(752, 480,
                435.2046959714599, 435.2046959714599,
                367.4517211914062,252.2008514404297);  //使用EuRoc,yaml中的parameter
       // cam_ = new svo::PinholeCamera(752, 480, 458.654, 457.296, 367.215, 248.375);//from sensor.yaml

        //vo_ = new svo::FrameHandlerMono(cam_);
        vo_ = new svo::FrameHandlerMono(cam_, dataset);

        vo_->start();

        viewer_ = new SLAM_VIEWER::Viewer(vo_);
        viewer_thread_ = new std::thread(&SLAM_VIEWER::Viewer::run, viewer_);
        viewer_thread_->detach();

        vioinit_ = new svo::VioInitialization(vo_);
        vioinit_thread_ = new std::thread(&svo::VioInitialization::run, vioinit_);
        vioinit_thread_->detach();

        IMUData::setGyrBiasRW2_Cov(
                dataset->imu_params[0].gyroscope_random_walk);       //gyroscope_random_walk: 1.9393e-05
        IMUData::setGyrMeasCov(
                dataset->imu_params[0].gyroscope_noise_density);     //gyroscope_noise_density: 1.6968e-04
        IMUData::setAccBiasRW2_Cov(
                dataset->imu_params[0].accelerometer_random_walk);   //accelerometer_random_walk: 3.0000e-3
        IMUData::setAccMeasCov(
                dataset->imu_params[0].accelerometer_noise_density); //accelerometer_noise_density: 2.0000e-3

    }

    BenchmarkNode::~BenchmarkNode() {
        delete vo_;
        delete cam_;
//        delete cam_r_;
//        delete cam_pinhole_;

        delete viewer_;
        delete viewer_thread_;

        delete vioinit_;
        delete vioinit_thread_;
    }

    void BenchmarkNode::runFromFolder() {

        for(int img_id = Config::StartFrame(); img_id < (int)dataset->img_timestamps[0].size(); ++img_id){

            // 1.读取左图像数据
            std::stringstream ss;  //ss左图像路经：　数据集路经/cam0/data/timestamps.png
            ss << dataset->cam_data_files[0][0] << dataset->img_timestamps[0][img_id] << ".png";
            cv::Mat img_left(cv::imread(ss.str().c_str(), CV_LOAD_IMAGE_UNCHANGED));
            assert(!img_left.empty());

            // 读取右图像数据
            std::stringstream ssr;  //ss左图像路经：　数据集路经/cam0/data/timestamps.png
            ssr << dataset->cam_data_files[1][0] << dataset->img_timestamps[0][img_id] << ".png";
            cv::Mat img_right(cv::imread(ssr.str().c_str(), CV_LOAD_IMAGE_UNCHANGED));
            assert(!img_right.empty());

            // 2.圖像畸变矫正
            DistortImg(img_left, 0);
            DistortImg(img_right, 1);

            // 3.图像处理与跟踪定位
//            vo_->addImage(img_left, GetRealTime( dataset->img_timestamps[0][img_id] )); //改了一下timestamp
//            cout<< "读取左图像数据:" << dataset->cam_data_files[1][0] << dataset->img_timestamps[0][img_id] <<endl;
            vo_->addImage(img_right, GetRealTime( dataset->img_timestamps[0][img_id] ));
            cout<< "读取右图像数据:" << dataset->cam_data_files[1][0] << dataset->img_timestamps[0][img_id] <<endl;
            vo_->set_Image_id(img_id);

            // 4.位姿结果输出
            OutPutFile( vo_, dataset, img_id);

            #ifdef USE_IMU   //#define USE_IMU in global.h
            //TODO:1.IMU input add here
            while(1){

                static int cur_imu_id = 1;
                static double last_img_timestamp = GetRealTime(dataset->img_timestamps[0][img_id]);

                //循环到最后一张图片时，跳出
                if(img_id == (int)( dataset->img_timestamps[0].size() -1 )){
                    break;
                }
                double next_img_timestamp = GetRealTime(dataset->img_timestamps[0][img_id + 1]);

                static double last_imu_timestamp = GetRealTime(dataset->imu_time_wa[0][cur_imu_id-1].first);
                static double cur_imu_timestamp = GetRealTime(dataset->imu_time_wa[0][cur_imu_id].first);

                //循环到最后一个imu数据时，跳出
                if (cur_imu_id == (int)(dataset->imu_time_wa[0].size() - 1))
                {
                    break;
                }
                double next_imu_timestamp = GetRealTime(dataset->imu_time_wa[0][cur_imu_id + 1].first);

                ///分段读取相邻两图像间的　imu数据
                /// 1.check head(only in the initial situation)
                if (cur_imu_timestamp < last_img_timestamp)   //clear all imu data before last_img_timestamp
                {
                    cur_imu_id++;
                    // last_img_timestampe = cur_imu_timestamp;
                    last_imu_timestamp = cur_imu_timestamp;
                    cur_imu_timestamp = next_imu_timestamp;
                    continue;
                }
                if(last_imu_timestamp < last_img_timestamp && cur_imu_timestamp >= last_img_timestamp){
                    // IMUData(const Eigen::Matrix<double, 6, 1>& g_a,const double&& t)  //delta t here
                    IMUData imudata( dataset->imu_time_wa[0][cur_imu_id - 1].second,
                                     cur_imu_timestamp - last_img_timestamp);
                    vo_->addImu(imudata);
                }

                /// 2.check tail
                if(next_imu_timestamp >= next_img_timestamp) {
                    IMUData imudata( dataset->imu_time_wa[0][cur_imu_id].second,
                                     next_img_timestamp - cur_imu_timestamp);
                    vo_->addImu(imudata);

                    last_imu_timestamp = cur_imu_timestamp;
                    cur_imu_timestamp = next_imu_timestamp;
                    last_img_timestamp = next_img_timestamp;
                    cur_imu_id++;
                    break;
                }
                /// 3. integrate from head to tail
                IMUData imudata( dataset->imu_time_wa[0][cur_imu_id].second,
                                 next_imu_timestamp - cur_imu_timestamp );
                vo_->addImu(imudata);

                last_imu_timestamp = cur_imu_timestamp;
                cur_imu_timestamp = next_imu_timestamp;
                cur_imu_id++;

            }
            usleep(100);
            #endif
        }

    }

}// namespace svo

int main(int argc, char** argv)
{

    svo::BenchmarkNode benchmark;
    benchmark.runFromFolder();

  printf("BenchmarkNode finished.\n");
  return 0;
}

