//
// Created by gzh on 20-4-14.
//

#ifndef SVO_EDGELETE_LIVE_DATA_EUROC_H
#define SVO_EDGELETE_LIVE_DATA_EUROC_H

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/StdVector>
#include "global.h"

using namespace std;
using namespace Eigen;

struct IMUParam{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix4d T_BS;

    int rate_hz;
    double gyroscope_noise_density;
    double gyroscope_random_walk;
    double accelerometer_noise_density;
    double accelerometer_random_walk;
};

struct CameraParam
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix4d T_BS;
    cv::Mat T_BC;
    int rate_hz;
    Vector2i resolution;
    string camera_model;
    Vector4d intrinsics;
    string distortion_model;
    Vector4d distortion_coefficients;
};

class EuRocData{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EuRocData(string dir, int cams=1, int imus=1);
    void readImgsAndParams();

    void readImusAndParams();

    void readGroundTruth();

    void readConfigParam();

private:
    const string body_yaml = "body.yaml";
    const string sensor_yaml = "sensor.yaml";
    const string data_csv = "data.csv"; //相机參數

    string mav_filedir;  //数据集路经
    int cam_num;         //图像 文件个数
    int imu_num;         //imu 文件个数

public:
    vector<string> cam_data_files;         //保存图像文件路经：cam_data_file　保存：数据集路经/cam0/data/
    vector<vector<string>> img_timestamps; //cam_vec 每个图像的时间戳

    vector< vector< pair<string,Matrix<double, 6, 1>>>> imu_time_wa; //保存每组imu数据的 (时间戳 ,６＊１的 (w & a) )

    vector<CameraParam> cam_params;
    vector<IMUParam> imu_params;
    Vector3d gravity;

    //获得真实位姿groundtruth
    vector < pair<string, Eigen::Vector3d>> ground_truth_p;
    vector < pair<string, Eigen::Quaterniond>> ground_truth_q;
    Vector3d getTruthTranslation(int id){
        return ground_truth_p[id].second;
    }
    Quaterniond getTrueQuaternion(int id){
        return ground_truth_q[id].second;
    }

};
#endif //SVO_EDGELETE_LIVE_DATA_EUROC_H
