//
// Created by gzh on 20-4-14.
//
#include "svo/Data_EuRoc.h"
#include "svo/Converter.h"
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

EuRocData::EuRocData(string dir, int cams, int imus) :    // EuRocData 初始化列表
            mav_filedir(dir), cam_num(cams), imu_num(imus)
{
    gravity<<  0.0 , 0.0 , 9.8012 ;
    readImgsAndParams();  // 获取数据集的图像信息
    readImusAndParams();  // 获取 imu 数据
    readGroundTruth();    // 获取 groundtruth 数据
}

void EuRocData::readImgsAndParams() {
cout<<"readImgsAndParams"<<endl;
    for(int i=0; i<cam_num; ++i){
        vector<string> cam_vec;
        string cam_= mav_filedir + "/"+ "cam"+ to_string(i)+ "/";  ///  数据集路经/ cam0/
        ifstream cam_file(cam_ +data_csv);           ///读取图像时间戳：cam_file ->数据集路经/cam0/data_csv

        string cam_data_file = cam_ + "data/";       ///保存图像文件路经：cam_data_file　保存：数据集路经/cam0/data/
        cam_data_files.push_back(cam_data_file);

        if (!cam_file.good())
            cerr << "cam" << i << " csv file not found !" << endl;
        int num_of_imgs = 0;
        string cur_line;

        getline(cam_file, cur_line);  ///去除第一行说明
        while(getline(cam_file, cur_line, ',')){   //遇到' ,'　停止读入
            if(cur_line == "") break;
            num_of_imgs++;
            cam_vec.push_back(cur_line);   ///cam_vec== 每个图像的时间戳
            getline(cam_file,cur_line);
        }

        img_timestamps.push_back(cam_vec);
        cout <<"num_of_imgs is:\t"<< num_of_imgs << endl;

        ///获取相机参数文件中的信息
        cv::FileStorage param_file(cam_ + sensor_yaml, cv::FileStorage::READ);
        CameraParam cp;
        param_file["rate_hz"]>>cp.rate_hz;
        param_file["camera_model"] >> cp.camera_model;
        param_file["distortion_model"] >> cp.distortion_model;

        cv::FileNode T_BS_node = param_file["T_BS"]["data"];
        cv::FileNode resolution_node = param_file["resolution"];
        cv::FileNode intrinsics_node = param_file["intrinsics"];
        cv::FileNode distortion_coefficients_node = param_file["distortion_coefficients"];

        cp.T_BS<< T_BS_node[0], T_BS_node[1], T_BS_node[2], T_BS_node[3],
                T_BS_node[4], T_BS_node[5], T_BS_node[6], T_BS_node[7],
                T_BS_node[8], T_BS_node[9], T_BS_node[10], T_BS_node[11],
                T_BS_node[12], T_BS_node[13], T_BS_node[14], T_BS_node[15];
        cp.T_BC=svo::Converter::toCvMat(cp.T_BS);
        //cout<<" cp.T_BC="<< cp.T_BC<<endl;
        cp.resolution << resolution_node[0], resolution_node[1];
        cp.intrinsics << intrinsics_node[0], intrinsics_node[1], intrinsics_node[2], intrinsics_node[3];
        cp.distortion_coefficients << distortion_coefficients_node[0], distortion_coefficients_node[1],
                distortion_coefficients_node[2], distortion_coefficients_node[3];

        cam_params.push_back(cp);
    }
}
void EuRocData::readImusAndParams() {

    for(int i = 0; i < imu_num; ++i ){
        vector< pair<string,Eigen::Matrix<double, 6, 1>>> imu_pair_vec;
        pair<string,Eigen::Matrix<double, 6, 1>> imu_pair;
        // imu path
        string imu_ = mav_filedir + "/" + "imu" + to_string(i) + "/";  /// imu路经/ imu0/
        ifstream imu_file(imu_ + data_csv);    ///读取imu数据：　imu路经/ imu0/ data_csv

        if (!imu_file.good())
            cerr << "imu" << i << " csv file not found !" << endl;
        int num_of_imus = 0;
        string cur_line;

        getline(imu_file, cur_line); // first line delete
        while (getline(imu_file, cur_line, ',')) {

            if (cur_line == "") break;
            num_of_imus++;
            imu_pair.first = cur_line;
            for (int i = 0; i < 5; i++)        /// imu_pair== 每组imu数据为：６＊１的 (w & a)
            {
                getline(imu_file, cur_line, ',');
                imu_pair.second(i) = stod(cur_line);
            }
            getline(imu_file, cur_line);
            imu_pair.second(5) = stod(cur_line);
            imu_pair_vec.push_back(imu_pair);  /// imu_pair_vec== 每组imu数据的 (时间戳 ,６＊１的 (w & a) )
        }
        imu_time_wa.push_back(imu_pair_vec);
        cout <<"num_of_imus is:\t"<< num_of_imus << endl;

        ///获取 imu参数文件 中的信息
        cv::FileStorage param_file(imu_ + sensor_yaml, cv::FileStorage::READ);
        IMUParam ip;
        param_file["rate_hz"] >> ip.rate_hz;
        param_file["gyroscope_noise_density"] >> ip.gyroscope_noise_density;
        param_file["gyroscope_random_walk"] >> ip.gyroscope_random_walk;
        param_file["accelerometer_noise_density"] >> ip.accelerometer_noise_density;
        param_file["accelerometer_random_walk"] >> ip.accelerometer_random_walk;

        cv::FileNode T_BS_node = param_file["T_BS"]["data"];

        ip.T_BS << T_BS_node[0], T_BS_node[1], T_BS_node[2], T_BS_node[3],
                T_BS_node[4], T_BS_node[5], T_BS_node[6], T_BS_node[7],
                T_BS_node[8], T_BS_node[9], T_BS_node[10], T_BS_node[11],
                T_BS_node[12], T_BS_node[13], T_BS_node[14], T_BS_node[15];

        imu_params.push_back(ip);
    }
}

double GetRealTime(const string str){
    return stod(str)/1e9;
}

void EuRocData::readGroundTruth() {

    pair<string, Eigen::Vector3d> groundtruth_pair_p;
    pair<string, Eigen::Quaterniond> groundtruth_pair_q;
    pair<string, Eigen::Vector3d> groundtruth_last_p;
    pair<string, Eigen::Quaterniond> groundtruth_last_q;

    string groundtruth_= mav_filedir + "/" + "state_groundtruth_estimate0" + "/";   /// groundtruth_==路经/ state_groundtruth_estimate0/
    ifstream groundtruth_file(groundtruth_ + data_csv);

    if (!groundtruth_file.good())
        cerr << "ground_truth_ csv file not found !" << endl;
    int num_of_groundtruth = 0;
    string cur_line;
    getline(groundtruth_file, cur_line); // first line delete
    vector< double > q_coefficient;

    while (getline(groundtruth_file, cur_line, ',')){
        if (cur_line == "") break;
        num_of_groundtruth++;
        groundtruth_pair_p.first = cur_line;
        groundtruth_pair_q.first = cur_line;
        for (int i = 0; i < 3; i++)
        {
            getline(groundtruth_file, cur_line, ',');
            groundtruth_pair_p.second(i) = stod(cur_line);
        }
        for (int i = 0; i < 4; i++)
        {
            getline(groundtruth_file, cur_line, ',');
            q_coefficient.push_back(stod(cur_line));
        }
        groundtruth_pair_q.second = Quaterniond(q_coefficient[1],q_coefficient[2],q_coefficient[3],q_coefficient[0]) ; //save as Quaterniond(x,y,z,w)
        q_coefficient.clear();

        for (int i = 0; i < 8; i++)
        {
            getline(groundtruth_file, cur_line, ',');
        }
        getline(groundtruth_file, cur_line);

        double img_time = GetRealTime(img_timestamps[0][ground_truth_p.size()]);
        double groundtruth_time = GetRealTime(groundtruth_pair_p.first);

        while(num_of_groundtruth <=1 && (groundtruth_time >=img_time)){
            ground_truth_p.push_back(groundtruth_pair_p);
            ground_truth_q.push_back(groundtruth_pair_q);
            img_time = GetRealTime(img_timestamps[0][ground_truth_p.size()]);
        }

        if((groundtruth_time >= img_time) && (num_of_groundtruth > 1)){
            double groundtruth_last_time = GetRealTime(groundtruth_last_p.first);
            if(groundtruth_last_time <= img_time){
                //double img_last_time=GetRealTime(img_timestamps[0][num_of_ground_truth-1]);
                if((groundtruth_time - img_time)<=(img_time - groundtruth_last_time)){
                    ground_truth_p.push_back(groundtruth_pair_p);
                    ground_truth_q.push_back(groundtruth_pair_q);
                }
                else{
                    ground_truth_p.push_back(groundtruth_last_p);
                    ground_truth_q.push_back(groundtruth_last_q);
                }
            }else {

//            ground_truth_last=ground_truth_pair;
//            continue;
            }
        }else{

        }
        groundtruth_last_p = groundtruth_pair_p;
        groundtruth_last_q = groundtruth_pair_q;
        //imu_timestamps.push_back(ground_truth_pair_vec);

    }

}


