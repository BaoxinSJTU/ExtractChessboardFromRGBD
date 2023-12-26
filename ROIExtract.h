#ifndef ROIEXTRACT_H
#define ROIEXTRACT_H
#include<open3d/Open3D.h>
#include<opencv4/opencv2/opencv.hpp>
#include<spdlog/spdlog.h>
class ROIExtract{
public:
    ROIExtract(){
        open3d::io::ReadIJsonConvertibleFromJSON("../rs-cfg.json", rs_cfg);
        rs.InitSensor(rs_cfg, 0, "1.bag");
        sensor_intrinsic = open3d::core::eigen_converter::EigenMatrixToTensor(rs.GetMetadata().intrinsics_.intrinsic_matrix_);

        depth_image = std::shared_ptr<open3d::geometry::Image>(new open3d::geometry::Image);
        color_image = std::shared_ptr<open3d::geometry::Image>(new open3d::geometry::Image);
        pcd = std::shared_ptr<open3d::geometry::PointCloud>(new open3d::geometry::PointCloud);

        main_process = std::thread(&ROIExtract::MainProcess, this);
    }
    ~ROIExtract(){
        if(main_process.joinable())
            main_process.join();
    }
    cv::Mat open3d2cv(const open3d::geometry::Image& src){
        int type = src.num_of_channels_ == 1? CV_16UC1 : CV_8UC3;
        return cv::Mat(src.height_, src.width_, type, (void*)src.data_.data());
    }
    open3d::geometry::Image cv2open3d(const cv::Mat& src){
        open3d::geometry::Image ret;
        int bytes_per_channel = src.depth()/2+1;
        ret.Prepare(src.cols, src.rows, src.channels(), bytes_per_channel);

        std::memcpy(ret.data_.data(), src.data, src.total() * src.channels() * bytes_per_channel);

        return ret;
    }
    void fill_chessboard(cv::Mat& src, std::vector<cv::Point2f>& corners){

        std::vector<cv::Point2f> poly1{corners.at(0), corners.at(1), corners.at(8), corners.at(7)};
        std::vector<cv::Point2f> poly2{corners.at(2), corners.at(3), corners.at(10), corners.at(9)};
        std::vector<cv::Point2f> poly3{corners.at(4), corners.at(5), corners.at(12), corners.at(11)};
        std::vector<cv::Point2f> poly4{corners.at(8), corners.at(9), corners.at(16), corners.at(15)};
        std::vector<cv::Point2f> poly5{corners.at(10), corners.at(11), corners.at(18), corners.at(17)};
        std::vector<cv::Point2f> poly6{corners.at(12), corners.at(13), corners.at(20), corners.at(19)};
        std::vector<cv::Point2f> poly7{corners.at(14), corners.at(15), corners.at(22), corners.at(21)};
        std::vector<cv::Point2f> poly8{corners.at(16), corners.at(17), corners.at(24), corners.at(23)};
        std::vector<cv::Point2f> poly9{corners.at(18), corners.at(19), corners.at(26), corners.at(25)};
        std::vector<cv::Point2f> poly10{corners.at(22), corners.at(23), corners.at(30), corners.at(29)};
        std::vector<cv::Point2f> poly11{corners.at(24), corners.at(25), corners.at(32), corners.at(31)};
        std::vector<cv::Point2f> poly12{corners.at(26), corners.at(27), corners.at(34), corners.at(33)};
        std::vector<cv::Point2f> poly13{corners.at(28), corners.at(29), corners.at(36), corners.at(35)};
        std::vector<cv::Point2f> poly14{corners.at(30), corners.at(31), corners.at(38), corners.at(37)};
        std::vector<cv::Point2f> poly15{corners.at(32), corners.at(33), corners.at(40), corners.at(39)};

        std::vector<std::vector<cv::Point2f>> polys{poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8, poly9, poly10, poly11, poly12, poly13, poly14, poly15};
        for(int y = 0;y < src.rows; y++){
            for(int x = 0; x < src.cols; x++){
                cv::Point2f p(x, y);
                bool found = false;
                for(int i = 0;i < 15;i++){
                    if(cv::pointPolygonTest(polys.at(i), p, false) >= 0){
                        cv::Vec3b* row = src.ptr<cv::Vec3b>(y);
                        row[x] = cv::Vec3b(color_vec.at(i)[0], color_vec.at(i)[1], color_vec.at(i)[2]);
                        break;
                    }
                }
            }
        }
    }
    bool target_detection(cv::Mat& src, cv::Mat& mask){
        cv::Size pattern_size(7, 4);
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool pattern_found = cv::findChessboardCorners(gray, cv::Size(chessboard_pattern_width, chessboard_pattern_height), corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        if(pattern_found){
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
        }
        else{
            spdlog::error("can not found chessboard corners!");
            return false;
        }

        

        std::vector<cv::Point2f> corners_bounding;
        corners_bounding.push_back(corners.at(0));
        corners_bounding.push_back(corners.at(chessboard_pattern_width - 1));
        corners_bounding.push_back(corners.at(chessboard_pattern_width*(chessboard_pattern_height-1) + chessboard_pattern_width - 1));
        corners_bounding.push_back(corners.at(chessboard_pattern_width*(chessboard_pattern_height-1)));

        mask = cv::Mat::zeros(src.size(), CV_8UC1);
        std::vector<cv::Point> polyPoints;
        for(const auto& corner : corners_bounding)
            polyPoints.push_back(corner);
        cv::fillConvexPoly(mask, polyPoints.data(), polyPoints.size(), cv::Scalar(255));

        cv::drawChessboardCorners(src, cv::Size(7, 6), cv::Mat(corners), pattern_found);

        fill_chessboard(src, corners);

        return true;
    }
    void MaskDepth(open3d::t::geometry::Image& depth, cv::Mat& mask){
    void* data_raw = depth.GetDataPtr();
        if(depth.GetDtype() == open3d::core::Dtype::UInt16){
            spdlog::info("depth data format is uint_16!");

            uint16_t* data = reinterpret_cast<uint16_t*> (data_raw);
            for(int y = 0; y < depth.GetRows(); y++){
                for(int x = 0; x < depth.GetCols(); x++){
                    if(mask.at<uchar>(y, x) == uint8_t(0))
                        *(data + y * depth.GetCols() + x) = uint16_t(0);
                }
            }
        }
        else{
            spdlog::info("depth data format is not uint_16!");
        }
    }
    void MainProcess(){
        rs.StartCapture(false);
        while(true){
            auto rgbd_t = rs.CaptureFrame(true, true);
            cv::Mat temp_cv, mask;

            auto rgbd = rgbd_t.ToLegacy();
            depth_image = std::shared_ptr<open3d::geometry::Image>(&rgbd.depth_, [](open3d::geometry::Image*){});
            color_image = std::shared_ptr<open3d::geometry::Image>(&rgbd.color_, [](open3d::geometry::Image*){});

            temp_cv = open3d2cv(*color_image);
            if(!target_detection(temp_cv, mask))
                continue;
            MaskDepth(rgbd_t.depth_, mask);
            auto pcd_t = open3d::t::geometry::PointCloud::CreateFromRGBDImage(rgbd_t, sensor_intrinsic, open3d::core::Tensor::Eye(4, open3d::core::Float32, ((open3d::core::Device)("CPU:0"))),
                                                                            1000.f, 15.0f);
            auto pcd_ = pcd_t.ToLegacy();
            pcd = std::shared_ptr<open3d::geometry::PointCloud>(&pcd_, [](open3d::geometry::PointCloud*){});
            break;
        }
        rs.StopCapture();
        // open3d::visualization::DrawGeometries({pcd});
        open3d::io::WritePointCloudOption opt;
        open3d::io::WritePointCloud("../data/pcd/1.pcd", *pcd, opt);
    }
private:
    std::thread main_process;
    int chessboard_pattern_width{7};
    int chessboard_pattern_height{6};
    open3d::t::io::RealSenseSensor rs;
    open3d::t::io::RealSenseSensorConfig rs_cfg;
    open3d::core::Tensor sensor_intrinsic;

    std::shared_ptr<open3d::geometry::Image> depth_image;
    std::shared_ptr<open3d::geometry::Image> color_image;
    std::shared_ptr<open3d::geometry::PointCloud> pcd;
    std::vector<cv::Scalar> color_vec{
        cv::Scalar(255, 0, 0),
        cv::Scalar(135, 206, 235),
        cv::Scalar(255, 215, 0),
        cv::Scalar(0, 128, 128),
        cv::Scalar(255, 165, 0),
        cv::Scalar(230, 230, 250),
        cv::Scalar(220, 20, 60),
        cv::Scalar(152, 255, 152),
        cv::Scalar(255, 191, 0),
        cv::Scalar(173, 216, 230),
        cv::Scalar(255, 127, 80),
        cv::Scalar(64, 224, 208),
        cv::Scalar(255, 0, 255),
        cv::Scalar(112, 128, 144),
        cv::Scalar(253, 94, 83),
    };
};
#endif