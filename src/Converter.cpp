//
// Created by gzh on 20-4-14.
//
#include "svo/Converter.h"
namespace svo {

//    void Converter::updateNS(NavState &ns, const IMUPreintegrator &imupreint, const Vector3d &gw) {
//        Matrix3d dR = imupreint.getDeltaR();
//        Vector3d dP = imupreint.getDeltaP();
//        Vector3d dV = imupreint.getDeltaV();
//        double dt = imupreint.getDeltaTime();
//
//        Vector3d Pwbpre = ns.Get_P();
//        Matrix3d Rwbpre = ns.Get_RotMatrix();
//        Vector3d Vwbpre = ns.Get_V();
//        //std::cout<<"________________333333333333333333____________________"<<std::endl;
//        std::cout << "Pwbpre:" << Pwbpre << std::endl;
//        std::cout << "Rwbpre:" << Rwbpre << std::endl;
//        std::cout << "Vwbpre:" << Vwbpre << std::endl;
//
//        Matrix3d Rwb = Rwbpre * dR;
//        Vector3d Pwb = Pwbpre + Vwbpre * dt + 0.5 * gw * dt * dt + Rwbpre * dP;
//        Vector3d Vwb = Vwbpre + gw * dt + Rwbpre * dV;
//
//        // Here assume that the pre-integration is re-computed after bias updated, so the bias term is ignored
//        ns.Set_Pos(Pwb);
//        ns.Set_Vel(Vwb);
//        ns.Set_Rot(Rwb);
//
//        // Test log
//        if (ns.Get_dBias_Gyr().norm() > 1e-6 || ns.Get_dBias_Acc().norm() > 1e-6)
//            std::cerr << "delta bias in updateNS is not zero" << ns.Get_dBias_Gyr().transpose() << ", "
//                      << ns.Get_dBias_Acc().transpose() << std::endl;
//    }

    cv::Mat Converter::toCvMatInverse(const cv::Mat &Tcw) {
        cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
        cv::Mat Rwc = Rcw.t();
        cv::Mat twc = -Rwc * tcw;

        cv::Mat Twc = cv::Mat::eye(4, 4, Tcw.type());
        Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
        twc.copyTo(Twc.rowRange(0, 3).col(3));

        return Twc.clone();
    }

    std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors) {
        std::vector<cv::Mat> vDesc;
        vDesc.reserve(Descriptors.rows);
        for (int j = 0; j < Descriptors.rows; j++)
            vDesc.push_back(Descriptors.row(j));

        return vDesc;
    }

    g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT) {
        Eigen::Matrix<double, 3, 3> R;
        R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
                cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
                cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

        Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

        return g2o::SE3Quat(R, t);
    }

    cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3) {
        Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
        return toCvMat(eigMat);
    }


    cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3) {
        Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = Sim3.translation();
        double s = Sim3.scale();
        return toCvSE3(s * eigR, eigt);
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m) {
        cv::Mat cvMat(4, 4, CV_32F);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cvMat.at<float>(i, j) = m(i, j);

        return cvMat.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m) {
        cv::Mat cvMat(3, 3, CV_32F);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cvMat.at<float>(i, j) = m(i, j);

        return cvMat.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1> &m) {
        cv::Mat cvMat(3, 1, CV_32F);
        for (int i = 0; i < 3; i++)
            cvMat.at<float>(i) = m(i);

        return cvMat.clone();
    }

    cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t) {
        cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cvMat.at<float>(i, j) = R(i, j);
            }
        }
        for (int i = 0; i < 3; i++) {
            cvMat.at<float>(i, 3) = t(i);
        }

        return cvMat.clone();
    }

    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVector) {
        Eigen::Matrix<double, 3, 1> v;
        v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

        return v;
    }

    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f &cvPoint) {
        Eigen::Matrix<double, 3, 1> v;
        v << cvPoint.x, cvPoint.y, cvPoint.z;

        return v;
    }

    Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat &cvMat3) {
        Eigen::Matrix<double, 3, 3> M;

        M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
                cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
                cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);

        return M;
    }

    Eigen::Matrix<double, 4, 4> Converter::toMatrix4d(const cv::Mat &cvMat4) {
        Eigen::Matrix<double, 4, 4> M;

        M << cvMat4.at<float>(0, 0), cvMat4.at<float>(0, 1), cvMat4.at<float>(0, 2), cvMat4.at<float>(0, 3),
                cvMat4.at<float>(1, 0), cvMat4.at<float>(1, 1), cvMat4.at<float>(1, 2), cvMat4.at<float>(1, 3),
                cvMat4.at<float>(2, 0), cvMat4.at<float>(2, 1), cvMat4.at<float>(2, 2), cvMat4.at<float>(2, 3),
                cvMat4.at<float>(3, 0), cvMat4.at<float>(3, 1), cvMat4.at<float>(3, 2), cvMat4.at<float>(3, 3);

        return M;
    }

    std::vector<float> Converter::toQuaternion(const cv::Mat &M) {
        Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
        Eigen::Quaterniond q(eigMat);

        std::vector<float> v(4);
        v[0] = q.x();
        v[1] = q.y();
        v[2] = q.z();
        v[3] = q.w();

        return v;
    }

}


