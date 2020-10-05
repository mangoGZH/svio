//
// Created by gzh on 20-4-16.
//

#include <svo/vio_init.h>

#ifdef  USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#include <svo/depth_filter.h>
#include <svo/config.h>
#endif

using namespace std;
using namespace cv;
namespace svo{

    VioInitialization::VioInitialization(svo::FrameHandlerMono *vo) :  gzh(1), lx(1), _vo(vo) {
        count =0;
        s_convergence_time =0;
        last_kf_id =0;
        FirstTry =true;
    }

    void VioInitialization::run() {

        bool init_finish = false;
#ifdef USE_IMU
        while( gzh && lx ) {  //此处循环需要一定的时间延时  usleep(100)/  cout
            int map_size = _vo->map().keyframes_.size();//map().size();
           //cout<<"map_size="<<map_size<<endl;
           // usleep(100);
            if( map_size > 1){
                int cur_kf_id = _vo->getMap()->lastKeyframe().get()->kf_id_;
                if( cur_kf_id >= 5) {
                    if (init_finish) {
                        break;
                    } else {
//                        cout<<"cur_kf_id="<<cur_kf_id<<endl;
//                        cout<<"last_kf_id="<<last_kf_id<<endl;
                        if (cur_kf_id > last_kf_id) {
                            cout << "*****kf is update ,try vio now****" << endl;
                            init_finish = TryVioInit();
                            last_kf_id = cur_kf_id;
                        } else {
                            //cout << "*****error: cur_kf_id <= last_kf_id****" << endl;
                        }
                    }
                }
                usleep(1000);  // sleep: 1ms (1000 * 10e-6 s)
            }
        }
#endif
    }

    void VioInitialization::Record_file(){

        static bool fopened = false;
        static ofstream gw,scale,biasa,condnum,time,biasg;
        string  filepath = "/home/gzh/vio_init_outfile/";
        if(!fopened)
        {
            gw.open(filepath+"gw.txt");
            scale.open(filepath+"scale.txt");
            biasa.open(filepath+"biasa.txt");
            condnum.open(filepath+"condnum.txt");
            time.open(filepath+"computetime.txt");
            biasg.open(filepath+"biasg.txt");
            if(gw.is_open() && scale.is_open() && biasa.is_open() &&
               condnum.is_open() && time.is_open() && biasg.is_open())
                fopened = true;
            else
            {
                cerr<<"file open error in TryInitVIO"<<endl;
                fopened = false;
            }
            gw<<std::fixed<<std::setprecision(6);
            scale<<std::fixed<<std::setprecision(6);
            biasa<<std::fixed<<std::setprecision(6);
            condnum<<std::fixed<<std::setprecision(6);
            time<<std::fixed<<std::setprecision(6);
            biasg<<std::fixed<<std::setprecision(6);
        }

    }

    bool VioInitialization::TryVioInit() {

        while( !_vo->getPermission_Read_kf()){  //等keyframe更新完毕，获得读取kf许可
            usleep(500);
        }
        // openfile: gw,scale,ba,bg  记录联合初始化参数估计结果
        //Record_file();

        Map* mp = _vo->getMap();
        size_t N = _vo->map().size();
        list< FramePtr> All_kf = mp->getAllKeyframe();
        auto endkf = All_kf.end();

        //ba::globalBA( mp );

        // Extrinsics  4*4  [body_frame--camera_frame]
        cv::Mat Tbc = _vo->getParam()->cam_params[0].T_BC;
        cv::Mat Rbc = Tbc.rowRange(0, 3).colRange(0, 3);
        cv::Mat pbc = Tbc.rowRange(0, 3).col(3);

        cv::Mat Rcb = Rbc.t();
        cv::Mat pcb = -Rcb * pbc;

        vector<Mat> Pose_c_w;  //Pose c to w
        vector<IMUPreintegrator> IMUPreInt;
        vector<KeyFrameInit* > kfInit;

        int cnt=0;
        for(auto it = All_kf.begin(); it != All_kf.end(); ++it, ++cnt){
            Frame* pkf = it->get();
            Pose_c_w.push_back( pkf->GetVPoseInverse()); //将每个kf的转移矩阵T 从相机系ｃ变换到世界系ｗ

            IMUPreInt.push_back( pkf->kfimuPreint);
            KeyFrameInit* kf = new KeyFrameInit(*pkf);

            if(cnt != 0){
                kf->prev_KeyFrame = kfInit[cnt - 1];  //存入上一帧KeyFrameInit
            }
            kfInit.push_back(kf); //存入当前帧KeyFrameInit
        }

        ///Step 1. ****estimate: bg*****
        Vector3d bgest = VioInitialization::solveGyroscopeBias( Pose_c_w, IMUPreInt, _vo->getParam()->cam_params[0].T_BS);
        //cout<<"bgest = "<<bgest<<"\t"<<endl;

        // Update biasg and pre-integration in LocalWindow. Remember to reset back to zero
        for (int i = 0; i < (int)N; i++) {
            kfInit[i]->bg = bgest;
        }
        for (int i = 0; i < (int)N; i++) {
            kfInit[i]->ComputePreInt();    //preint with  bgest
        }

        /// Step 2.  ****estimate: scale_  gw_ *****
        // Solve A*x=B for x=[s,gw] 4x1 vector
        cv::Mat A = cv::Mat::zeros(3 * (N - 2), 4, CV_32F);
        cv::Mat B = cv::Mat::zeros(3 * (N - 2), 1, CV_32F);
        cv::Mat I3 = cv::Mat::eye(3, 3, CV_32F);

        for(int i = 0; i < (int)N-2; ++i ){
            //KeyFrameInit* pKF1 = kfInit[i];
            KeyFrameInit *pKF2 = kfInit[i + 1];
            KeyFrameInit *pKF3 = kfInit[i + 2];

            // Delta time between frames
            double dt12 = pKF2->IMUPreInt.getDeltaTime();
            double dt23 = pKF3->IMUPreInt.getDeltaTime();

            // Pre-integrated measurements
            cv::Mat dp12 = Converter::toCvMat(pKF2->IMUPreInt.getDeltaP());
            cv::Mat dv12 = Converter::toCvMat(pKF2->IMUPreInt.getDeltaV());
            cv::Mat dp23 = Converter::toCvMat(pKF3->IMUPreInt.getDeltaP());

            // Pose of camera in world frame, Twc
            cv::Mat Twc1 = Pose_c_w[i].clone();//pKF1->GetPoseInverse();
            cv::Mat Twc2 = Pose_c_w[i + 1].clone();//pKF2->GetPoseInverse();
            cv::Mat Twc3 = Pose_c_w[i + 2].clone();//pKF3->GetPoseInverse();
            // Position of camera center, pwc
            cv::Mat pc1 = Twc1.rowRange(0, 3).col(3);
            cv::Mat pc2 = Twc2.rowRange(0, 3).col(3);
            cv::Mat pc3 = Twc3.rowRange(0, 3).col(3);
            // Rotation of camera, Rwc
            cv::Mat Rc1 = Twc1.rowRange(0, 3).colRange(0, 3);
            cv::Mat Rc2 = Twc2.rowRange(0, 3).colRange(0, 3);
            cv::Mat Rc3 = Twc3.rowRange(0, 3).colRange(0, 3);

            // lambda*s + beta*g = gamma
            cv::Mat lambda = (pc2 - pc1) * dt23 + (pc2 - pc3) * dt12;
            cv::Mat beta = 0.5 * I3 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23);
            cv::Mat gamma = (Rc3 - Rc2) * pcb * dt12 + (Rc1 - Rc2) * pcb * dt23 + Rc1 * Rcb * dp12 * dt23 -
                            Rc2 * Rcb * dp23 * dt12 - Rc1 * Rcb * dv12 * dt12 * dt23;
            lambda.copyTo(A.rowRange(3 * i + 0, 3 * i + 3).col(0));
            beta.copyTo(A.rowRange(3 * i + 0, 3 * i + 3).colRange(1, 4));
            gamma.copyTo(B.rowRange(3 * i + 0, 3 * i + 3));
            // 论文<<visua-inertail monocular slam with map reuse>> 使用 -gamma. Then the scale and gravity vector is -xx
        }
        // Use svd to compute A*x=B, x=[s,gw] 4x1 vector
        cv::Mat w, u, vt;                                // A = u*w*vt,  u*w*vt*x=B
        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A);    // Note w is 4x1 vector by SVDecomp()
//        cout<<"u:"<<u<<endl;
//        cout<<"vt:"<<vt<<endl;
//        cout<<"w:"<<w<<endl;

        //get condition number = 最大奇异值/最小奇异值
        double condition_num = w.at<float>(0)/w.at<float>(3);
//        cout<<"condition_num1:"<<condition_num<<endl;

        // Compute winv
        cv::Mat winv = cv::Mat::eye(4, 4, CV_32F);
        for (int i = 0; i < 4; i++) {
            if (fabs(w.at<float>(i)) < 1e-10) {
                w.at<float>(i) += 1e-10;
                cerr << "w(i) < 1e-10, w=" << endl << w << endl;
            }
            winv.at<float>(i, i) = 1. / w.at<float>(i);  // winv is 4x4 diagonal matrix
        }
        // Then x = vt'*winv*u'*B
        cv::Mat x = vt.t() * winv * u.t() * B;

        // x=[s,gw] 4x1 vector
        //cout<<"x:"<<x<<endl;
        double s_ = x.at<float>(0);    // scale should be positive
        cv::Mat gw_ = x.rowRange(1, 4);    // gravity should be about ~9.8
//        cout << "(in step2 ) s_:\t" << s_ <<"\t"<< endl;
//        cout << "(in step2 ) gw_:\t" << gw_.t() << ", |gw_|=" << cv::norm(gw_) <<"\t"<< endl;

        if (w.type() != I3.type() || u.type() != I3.type() || vt.type() != I3.type())
            cerr << "different mat type, I3,w,u,vt: " << I3.type() << "," << w.type() << "," << u.type() << ","
                 << vt.type() << endl;


        /// Step 3.  ****estimate: ba scale gw*****
        // Use gravity magnitude 9.8 as constraint
        // gI = [0;0;1], the normalized gravity vector in an inertial frame, NED type with no orientation.
        cv::Mat gI = cv::Mat::zeros(3, 1, CV_32F);
        gI.at<float>(2) = 1;
        cv::Mat gwn = gw_ / cv::norm(gw_);

        // v_hat = (gI x gw) / |gI x gw|
        cv::Mat gIxgwn = gI.cross(gwn);
        double norm_gIxgwn = cv::norm(gIxgwn);
        cv::Mat v_hat = gIxgwn / norm_gIxgwn;
        double theta = atan2( norm_gIxgwn, gI.dot(gwn));
        //cout<<"vhat: "<<v_hat<<", theta: "<<theta*180.0/M_PI<<endl;

        Eigen::Vector3d V_hat = Converter::toVector3d(v_hat);
        Eigen::Matrix3d RWI = Sophus::SO3::exp(V_hat * theta).matrix();
        cv::Mat Rwi = Converter::toCvMat(RWI);
        cv::Mat GI = gI * 9.8012;

        // Solve C*x=D for x=[s,dthetaxy,ba] (1+2+3)x1 vector
        cv::Mat C = cv::Mat::zeros(3 * (N - 2), 6, CV_32F);
        cv::Mat D = cv::Mat::zeros(3 * (N - 2), 1, CV_32F);

        for (int i = 0; i < (int) N - 2; i++) {
            KeyFrameInit* pKF1 = kfInit[i];
            KeyFrameInit *pKF2 = kfInit[i + 1];
            KeyFrameInit *pKF3 = kfInit[i + 2];
            // Delta time between frames
            double dt12 = pKF2->IMUPreInt.getDeltaTime();
            double dt23 = pKF3->IMUPreInt.getDeltaTime();
            // Pre-integrated measurements
            cv::Mat dp12 = Converter::toCvMat(pKF2->IMUPreInt.getDeltaP());
            cv::Mat dv12 = Converter::toCvMat(pKF2->IMUPreInt.getDeltaV());
            cv::Mat dp23 = Converter::toCvMat(pKF3->IMUPreInt.getDeltaP());
            cv::Mat Jpba12 = Converter::toCvMat(pKF2->IMUPreInt.getJPBiasa());
            cv::Mat Jvba12 = Converter::toCvMat(pKF2->IMUPreInt.getJVBiasa());
            cv::Mat Jpba23 = Converter::toCvMat(pKF3->IMUPreInt.getJPBiasa());
            // Pose of camera in world frame
            cv::Mat Twc1 = Pose_c_w[i].clone();    //pKF1->GetPoseInverse();
            cv::Mat Twc2 = Pose_c_w[i + 1].clone();//pKF2->GetPoseInverse();
            cv::Mat Twc3 = Pose_c_w[i + 2].clone();//pKF3->GetPoseInverse();
            // Position of camera center
            cv::Mat pc1 = Twc1.rowRange(0, 3).col(3);
            cv::Mat pc2 = Twc2.rowRange(0, 3).col(3);
            cv::Mat pc3 = Twc3.rowRange(0, 3).col(3);
            // Rotation of camera, Rwc
            cv::Mat Rc1 = Twc1.rowRange(0, 3).colRange(0, 3);
            cv::Mat Rc2 = Twc2.rowRange(0, 3).colRange(0, 3);
            cv::Mat Rc3 = Twc3.rowRange(0, 3).colRange(0, 3);

            // Stack to C/D matrix
            // lambda*s + phi*dthetaxy + zeta*ba = psi
            cv::Mat lambda = (pc2 - pc1) * dt23 + (pc2 - pc3) * dt12;
            cv::Mat phi = -0.5 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23) * Rwi *
                          SkewSymmetricMatrix(GI);  // note: this has a '-', different to paper
            cv::Mat zeta = Rc2 * Rcb * Jpba23 * dt12 + Rc1 * Rcb * Jvba12 * dt12 * dt23 - Rc1 * Rcb * Jpba12 * dt23;
            cv::Mat psi = (Rc1 - Rc2) * pcb * dt23 + Rc1 * Rcb * dp12 * dt23 - (Rc2 - Rc3) * pcb * dt12
                          - Rc2 * Rcb * dp23 * dt12 - Rc1 * Rcb * dv12 * dt23 * dt12 -
                          0.5 * Rwi * GI * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23); // note:  - paper
            lambda.copyTo(C.rowRange(3 * i + 0, 3 * i + 3).col(0));
            phi.colRange(0, 2).copyTo(C.rowRange(3 * i + 0, 3 * i + 3).colRange(1,3)); //only the first 2 columns, third term in dtheta is zero, here compute dthetaxy 2x1.
            zeta.copyTo(C.rowRange(3 * i + 0, 3 * i + 3).colRange(3, 6));
            psi.copyTo(D.rowRange(3 * i + 0, 3 * i + 3));

        }

        // Use svd to compute C*x=D, x=[s,dthetaxy,ba] 6x1 vector
        cv::Mat w2, u2, vt2;                              // C = u*w*vt, u*w*vt*x=D
        cv::SVDecomp(C, w2, u2, vt2, cv::SVD::MODIFY_A);  // Note w2 is 6x1 vector by SVDecomp()
//        cout<<"u2:"<<endl<<u2<<endl;
//        cout<<"vt2:"<<endl<<vt2<<endl;
//        cout<<"w2:"<<endl<<w2<<endl;

        //get condition number = 最大奇异值/最小奇异值
        double condition_num2 = w2.at<float>(0)/w2.at<float>(5);
        cout<<"condition_num2:"<<condition_num2<<endl;

        cv::Mat w2inv = cv::Mat::eye(6, 6, CV_32F);
        for (int i = 0; i < 6; i++) {
            if (fabs(w2.at<float>(i)) < 1e-10) {
                w2.at<float>(i) += 1e-10;
                cerr << "w2(i) < 1e-10, w=" << endl << w2 << endl;
            }
            w2inv.at<float>(i, i) = 1. / w2.at<float>(i); //w2inv is 6x6 diagonal matrix
        }
        // Then y = vt2'*winv2*u2'*D
        cv::Mat y = vt2.t() * w2inv * u2.t() * D;

        double s = y.at<float>(0);
        cv::Mat d_thetaxy = y.rowRange(1, 3);
        cv::Mat d_biasa = y.rowRange(3, 6);
        Vector3d d_Biasa = Converter::toVector3d(d_biasa);

        // d_theta = [dx;dy;0]
        cv::Mat d_theta = cv::Mat::zeros(3, 1, CV_32F);
        d_thetaxy.copyTo(d_theta.rowRange(0, 2));
        Eigen::Vector3d d_Theta = Converter::toVector3d(d_theta);

        // Rwi_ = Rwi*exp(d_Theta)
        Eigen::Matrix3d RWI_ = RWI * Sophus::SO3::exp(d_Theta).matrix();
        cv::Mat Rwi_ = Converter::toCvMat(RWI_);

        listscale.push_back( s );
        int scale_size = listscale.size();
        cout<<"push_back s :"<< s <<" into listscale\t"<<endl;

        ///*******Judging Initialization Stop Conditions********
        bool init_compelte = false;

        if(FirstTry){
            FirstTry = false;
            StartTime = mp->lastKeyframe()->timestamp_;
        }

       //TODO :judge the VIO_initial_finish condition use 15s
        Frame* newest_kf = All_kf.back().get();
        if (newest_kf->timestamp_ - StartTime >= Config::VIOStopTime()) {
            cout << "time >15s ,VIO initial  OK!!!!!!!!!!! :" << endl;
            init_compelte = true;
        } else {
            cout << "VIO initial time not arrive  !!!   the processing time is:" << newest_kf->timestamp_ - StartTime<< endl;
        }
       //TODO :judge the VIO_initial_finish condition use scale_threshould
        if(scale_size >=10) {
            double sum_scale = 0;
            for (int i = scale_size - 1; i >= scale_size - 10; i--) { sum_scale += listscale[i]; }
            double mean_scale = sum_scale / 10;
            double accum = 0.0;
            for (int i = scale_size - 1; i >= scale_size - 10; i--) { accum += (listscale[i] - mean_scale) *(listscale[i] - mean_scale);}
            double stdev = sqrt(accum / 10);

            if(stdev < thresh_1)
                count++;
            else
                count=0;

            if((count >= 5) && (fabs(listscale[scale_size-1]-listscale[scale_size-2])<thresh_2 ))
            {
                s_convergence_time =  newest_kf->timestamp_ - StartTime;
                cout << "VIO initial  OK before 15s !!!!!!!!!!!:\t it use :"<< s_convergence_time <<"s\n"<<endl;
                cout << "the timestamp is: " << newest_kf->timestamp_ /1e9 << "\t scale is:\t"<< s_<<"\n"<<endl;
                init_compelte = true;
            } else{
                cout << "VIO initial unfinished  !!!   the processing time is:" << newest_kf->timestamp_ - StartTime << endl;
            }
        }

        ////*****************************record file data: gw,scale,ba,bg*******************************
        {  //open file
            static bool fopened = false;
            static ofstream gw,scale,biasa,condnum,time,biasg,Rwi;  //ofstream fRwi(filepath+"Rwi.txt");
            string  filepath = "/home/gzh/vio_init_outfile/";
            if(!fopened)
            {
                gw.open(filepath+"gw.txt");
                scale.open(filepath+"scale.txt");
                biasa.open(filepath+"biasa.txt");
                condnum.open(filepath+"condnum.txt");
                time.open(filepath+"computetime.txt");
                biasg.open(filepath+"biasg.txt");
                Rwi.open(filepath+"Rwi.txt");

                if(gw.is_open() && scale.is_open() && biasa.is_open() &&
                   condnum.is_open() && time.is_open() && biasg.is_open() && Rwi.is_open())
                    fopened = true;
                else
                {
                    cerr<<"file open error in TryInitVIO"<<endl;
                    fopened = false;
                }
                gw<<std::fixed<<std::setprecision(6);
                scale<<std::fixed<<std::setprecision(6);
                biasa<<std::fixed<<std::setprecision(6);
                condnum<<std::fixed<<std::setprecision(6);
                time<<std::fixed<<std::setprecision(6);
                biasg<<std::fixed<<std::setprecision(6);
                Rwi<<std::fixed<<std::setprecision(6);
            }

            //output file: gw,scale,ba,bg
            //cv::Mat gwbefore = Rwi  * GI;
            cv::Mat gw_2 = gw_;
            cv::Mat gw_3 = Rwi_ * GI;

            gw << newest_kf->timestamp_<<" "
               << newest_kf->timestamp_ - StartTime<<" "
               << newest_kf->kf_id_<<" "
               <<gw_2.at<float>(0)<<" "<<gw_2.at<float>(1)<<" "<<gw_2.at<float>(2)<<" "
               <<gw_3.at<float>(0)<<" "<<gw_3.at<float>(1)<<" "<<gw_3.at<float>(2)<<" "<<endl;

            scale << newest_kf->timestamp_<<" "
                  << newest_kf->timestamp_ - StartTime<<" "
                  << newest_kf->kf_id_<<" "
                  << s_<<" "<< s<<" "
                  << s_convergence_time<<" "<<endl;

            biasa << newest_kf->timestamp_<<" "
                  << newest_kf->timestamp_ - StartTime<<" "
                  << newest_kf->kf_id_<<" "
                  << d_biasa.at<float>(0)<<" "<<d_biasa.at<float>(1)<<" "<<d_biasa.at<float>(2)<<" "<<endl;

            biasg << newest_kf->timestamp_<<" "
                  << newest_kf->timestamp_ - StartTime<<" "
                  << newest_kf->kf_id_<<" "
                  << bgest(0)<<" "<<bgest(1)<<" "<<bgest(2)<<" "<<endl;

//            condnum << newest_kf->timestamp_<<" "
//                    << newest_kf->timestamp_ - StartTime<<" "
//                    << newest_kf->kf_id_<<" "
//                    << w2.at<float>(0)<<" "<<w2.at<float>(1)<<" "<<w2.at<float>(2)<<" "<<w2.at<float>(3)<<" "
//                    << w2.at<float>(4)<<" "<<w2.at<float>(5)<<" "<<endl;

            Rwi <<RWI_(0,0)<<" "<<RWI_(0,1)<<" "<<RWI_(0,2)<<"\n"
                <<RWI_(1,0)<<" "<<RWI_(1,1)<<" "<<RWI_(1,2)<<"\n"
                <<RWI_(2,0)<<" "<<RWI_(2,1)<<" "<<RWI_(2,2)<<endl;
            Rwi.close();
        }
       /// ************************VIO initial complete!!!!  only set once********
        if(init_compelte) {   // Set NavState , scale and bias for all KeyFrames
            double scale = s;
            cout << "scale (VIO initial complete):" << scale << endl;

            if (!Config::VIOScale())
                scale = 1;
            _vo->setMapScale( scale );

            // gravity vector in world frame
            cv::Mat gw = Rwi_ * GI;
            cout << "gw (VIO initial complete):" << gw << endl;

            GravityVec = gw.clone();
            Vector3d gravity = Converter::toVector3d(GravityVec);
            _vo->setGravityVec(gravity);

//            while (!_vo->getPermission_Update_kf()) {
//                static int wait_time = 0;
//                cout << "VioInit complete, waitting for Uptate KF Permission: " << wait_time++ << "\n" << endl;
//                usleep(1000);
//            }
            _vo->setPermission_ProcessFrame(false); // 初始化位姿更新过程中，不能进行新帧处理
            cout << "setPermission_ProcessFrame(false)初始化位姿更新过程中，不能进行新帧处理" << endl;

        // updating pose via : s /gw /bg /ba
            {
                int count = 0;
                for (auto it = All_kf.begin(); it != All_kf.end(); ++it, count++) {
                    Frame *pKF = it->get();

                    // Position and rotation of visual SLAM TODO check the kf quality
                    cv::Mat wPc = pKF->GetVPoseInverse().rowRange(0, 3).col(3);            //Pwc
                    cv::Mat Rwc = pKF->GetVPoseInverse().rowRange(0, 3).colRange(0, 3);   //Rwc

                    // Set position and rotation of navstate
                    cv::Mat wPb = scale * wPc + Rwc * pcb;
                    pKF->SetNavStatePos(Converter::toVector3d(wPb));        //Pwb
                    pKF->SetNavStateRot(Converter::toMatrix3d(Rwc * Rcb));  //Rwb

                    // Update bias of Gyr & Acc
                    pKF->SetNavStateBiasGyr(bgest);
                    pKF->SetNavStateBiasAcc(d_Biasa);

                    // Set delta_bias to zero. (only updated during optimization)
                    pKF->SetNavStateDeltaBg(Eigen::Vector3d::Zero());
                    pKF->SetNavStateDeltaBa(Eigen::Vector3d::Zero());


                    /// Step 4.  ****compute velocity*****
                    if (pKF != All_kf.back().get()) {
                        Frame *pKFnext = pKF->GetNextKeyFrame();
                        if (!pKFnext)
                            cerr << "pKFnext is NULL, cnt=" << cnt << ", pKFnext:" << pKFnext << endl;

                        // IMU preint between pKF ~ pKFnext
                        const IMUPreintegrator& imupreint = pKFnext->kfimuPreint;

                        double dt = imupreint.getDeltaTime();                                       // deltaTime
                        cv::Mat dp = Converter::toCvMat(imupreint.getDeltaP());                     // deltaP
                        cv::Mat Jpba = Converter::toCvMat(imupreint.getJPBiasa());                  // J_deltaP_biasa
                        cv::Mat wPcnext = pKFnext->GetVPoseInverse().rowRange(0, 3).col(3);         // wPc next
                        cv::Mat Rwcnext = pKFnext->GetVPoseInverse().rowRange(0, 3).colRange(0, 3); // Rwc next

                        cv::Mat vel = -1. / dt * (scale * (wPc - wPcnext) + (Rwc - Rwcnext) * pcb +
                                                  Rwc * Rcb * (dp + Jpba * d_biasa) + 0.5 * gw * dt * dt);
                        Eigen::Vector3d Vel = Converter::toVector3d(vel);
                        pKF->SetNavStateVel(Vel);
                    }
                    else {  // deal with last kf
                        Frame *pKFprev = pKF->GetPrevKeyFrame();
                        if (!pKFprev) cerr << "pKFprev is NULL, cnt=" << cnt << endl;
                        const IMUPreintegrator &imupreint_prev_cur = pKF->kfimuPreint;

                        double dt = imupreint_prev_cur.getDeltaTime();
                        Eigen::Matrix3d Jvba = imupreint_prev_cur.getJVBiasa();
                        Eigen::Vector3d dv = imupreint_prev_cur.getDeltaV();

                        Eigen::Vector3d velpre = pKFprev->GetNavState().Get_V();
                        Eigen::Matrix3d rotpre = pKFprev->GetNavState().Get_RotMatrix();
                        Eigen::Vector3d Vel = velpre + gravity * dt + rotpre * (dv + Jvba * d_Biasa);
                        pKF->SetNavStateVel(Vel);
                    }
                }

                //Re-compute IMU pre-integration at last. Should after usage of pre-int measurements.
                for (auto it = All_kf.begin(), end = All_kf.end(); it != end; it++) {
                    Frame *pKF = it->get();
                    pKF->ComputePreInt();
                }

                cout<<"Update kf poses"<<endl;
                /// Update poses (multiply metric scale)    update visual pose and visual ponit position in 3d
                list<FramePtr> mpKeyFrames = mp->getAllKeyframe();
                for (auto it = mpKeyFrames.begin(), end = mpKeyFrames.end(); it != end; it++) {
                    Frame *pKF = it->get();
                    cv::Mat pcw = pKF->GetVPos()*scale;         //w到c的位移pcw ,*scale
                    pKF->SetVPos(Converter::toVector3d(pcw));
                    //pKF->SetVPose(pcw);
                }
                cout<<"Update points position"<<endl;
                ///update the points position which are converged[point_candidates_]
                MapPointCandidates::PointCandidateList mp_point = mp->point_candidates_.GetAllMapPoints();
                for(auto it = mp_point.begin(), ite = mp_point.end(); it != ite; ++it){
                    Point *pmp = it->first; //pair<Point*, Feature*> PointCandidate;
                    if(pmp == NULL)
                        continue;
                    pmp->pos_ *= scale;
                }
                cout<<"update the seeds parameter"<<endl;
                ///update the seeds parameter which are not converged
                int seed_count=0;
                DepthFilter* seed_depth_filter_= _vo->depthFilter();
                std::list<Seed, aligned_allocator<Seed>> seeds=seed_depth_filter_->getSeeds();
                for(auto sit = seeds.begin(), send = seeds.end(); sit != send; ++sit){
                    if( sit ->ftr->point == NULL)
                        continue;
                    sit->mu /= scale;       //mu = 1/depth_mean
                    sit->z_range /= scale;  //z_range = 1/depth_mean
                    seed_count++;
                }
                cout << endl << "... scale updated finish!![1 imustate/ 2 kf_pos / 3 mappoint_pos] ... " << endl << endl;
            }

            _vo->vioInitFinish = true;
            _vo->setPermission_ProcessFrame(true);
            cout<<"svioInitFinish  and  setProcessFramePermission"<<endl;
        }

        //TODO: add global BA after initialization


        for(int i=0; i< (int)N ; ++i){
            if(kfInit[i])
                delete kfInit[i];
        }
        return init_compelte;
    }


    cv::Mat VioInitialization::SkewSymmetricMatrix(const cv::Mat &v) {
        return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
                v.at<float>(2), 0, -v.at<float>(0),
                -v.at<float>(1), v.at<float>(0), 0);
    }

    void KeyFrameInit::ComputePreInt() {
        if (prev_KeyFrame == NULL) {
            return;
        } else {
            IMUPreInt.reset();// Reset pre-integrator first

            if (kfIMUData.empty())
                return;
            // remember to consider the gap between the last KF and the first IMU

            const IMUData &imu = kfIMUData.front();

            // integrate each imu
            for (size_t i = 0; i < kfIMUData.size(); i++) {
                const IMUData &imu = kfIMUData[i];
                // update pre-integrator
                IMUPreInt.update(imu._g - bg, imu._a, imu._t);    ///IMUPreInt: from last KF to this KF

//                cout<<"imu._g - bg"<<imu._g - bg<<endl;
//                cout<<"imu._a"<<imu._a<<endl;
            }
        }
    }


    Vector3d VioInitialization::solveGyroscopeBias(const vector<cv::Mat> &Pose_c_w, const vector<IMUPreintegrator> &ImuPreInt,
                                const Matrix4d T_bc) {
        int N = Pose_c_w.size();
        if (Pose_c_w.size() != ImuPreInt.size()) cerr << "vTwc.size()!=vImuPreInt.size()" << endl;
        Matrix3d Rcb = T_bc.topLeftCorner(3, 3).transpose();

        Matrix3d A;
        Vector3d b;
        Vector3d delta_bg;
        A.setZero();
        b.setZero();
        for (int i = 0; i < N; i++) {

            if (i == 0)  // Ignore the first KF
                continue;
            const cv::Mat &Twi = Pose_c_w[i - 1];    // pose of previous KF
            const cv::Mat &Twj = Pose_c_w[i];        // pose of this KF
            Matrix3d Rwci = Converter::toMatrix3d(Twi.rowRange(0, 3).colRange(0, 3));
            Matrix3d Rwcj = Converter::toMatrix3d(Twj.rowRange(0, 3).colRange(0, 3));
            Matrix3d Rwbi = Rwci * Rcb;
            Matrix3d Rwbj = Rwcj * Rcb;
            const IMUPreintegrator &imupreint = ImuPreInt[i];

            MatrixXd tmp_A(3, 3);
            tmp_A.setZero();
            VectorXd tmp_b(3);
            tmp_b.setZero();

            Eigen::Quaterniond q_ij(Rwbi.transpose() * Rwbj);
            Eigen::Quaterniond q_DeltaR(imupreint.getDeltaR());
            // tmp_A =J_R_bg/
            tmp_A = imupreint.getJRBiasg();//Matrix3d J_rPhi_dbg = M.getJRBiasg();  // jacobian of preintegrated rotation-angle to gyro bias i
            tmp_b = 2 * (q_DeltaR.inverse() * q_ij).vec();
            //tmp_A * delta_bg = tmp_b
            A += tmp_A.transpose() * tmp_A;
            b += tmp_A.transpose() * tmp_b;
        }
        //LDLT METHOD
        delta_bg = A.ldlt().solve(b);
        //cout << "gyroscope bias -----> delta_bg =  " << delta_bg.transpose();

        return delta_bg;
    }

}