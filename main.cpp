#include<open3d/Open3D.h>
#include<spdlog/spdlog.h>
using namespace open3d::t;
#include<opencv4/opencv2/opencv.hpp>
#define PATTERN_WIDTH 7
#define PATTERN_HEIGHT 6

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
void target_detection(cv::Mat& src, cv::Mat& chessboard_partial){
    cv::Size pattern_size(7, 4);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    bool pattern_found = cv::findChessboardCorners(gray, cv::Size(PATTERN_WIDTH, PATTERN_HEIGHT), corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    if(pattern_found){
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    }
    else{
        spdlog::error("can not found chessboard corners!");
        return;
    }

    

    std::vector<cv::Point2f> corners_bounding;
    corners_bounding.push_back(corners.at(0));
    corners_bounding.push_back(corners.at(PATTERN_WIDTH - 1));
    corners_bounding.push_back(corners.at(PATTERN_WIDTH*(PATTERN_HEIGHT-1) + PATTERN_WIDTH - 1));
    corners_bounding.push_back(corners.at(PATTERN_WIDTH*(PATTERN_HEIGHT-1)));

    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    std::vector<cv::Point> polyPoints;
    for(const auto& corner : corners_bounding)
        polyPoints.push_back(corner);
    cv::fillConvexPoly(mask, polyPoints.data(), polyPoints.size(), cv::Scalar(255));
    src.copyTo(chessboard_partial, mask);

    cv::drawChessboardCorners(src, cv::Size(7, 6), cv::Mat(corners), pattern_found);

    fill_chessboard(src, corners);
}

int main(int argc, char** argv){
    io::RealSenseSensor::ListDevices();
    io::RealSenseSensorConfig rs_cfg;

    open3d::core::Tensor sensor_intrinsic;
    open3d::io::ReadIJsonConvertibleFromJSON("../rs-cfg.json", rs_cfg);
    
    io::RealSenseSensor rs;
    rs.InitSensor(rs_cfg, 0, "1.bag");
    sensor_intrinsic = open3d::core::eigen_converter::EigenMatrixToTensor(rs.GetMetadata().intrinsics_.intrinsic_matrix_);

    bool flag_start = false, flag_record = flag_start, flag_exit = false;
    open3d::visualization::VisualizerWithKeyCallback depth_vis, color_vis, pcd_vis;
    auto callback_exit = [&](open3d::visualization::Visualizer *vis) {
        flag_exit = true;
        if (flag_start) {
            open3d::utility::LogInfo("Recording finished.");
        } else {
            open3d::utility::LogInfo("Nothing has been recorded.");
        }
        return false;
    };
    std::shared_ptr<open3d::geometry::Image> depth_image(new open3d::geometry::Image);
    std::shared_ptr<open3d::geometry::Image> color_image(new open3d::geometry::Image);
    std::shared_ptr<open3d::geometry::PointCloud> pcd(new open3d::geometry::PointCloud);

    int index = 0;

    auto callback_save = [&](open3d::visualization::Visualizer *vis){
        if(flag_start){
            open3d::io::WriteImage("../data/color/"+std::to_string(index)+".png", *color_image);
            open3d::io::WriteImage("../data/depth/"+std::to_string(index)+".png", *depth_image);
            open3d::io::WritePointCloudOption opt;
            open3d::io::WritePointCloud("../data/pcd/"+std::to_string(index)+".pcd", *pcd, opt);
            index++;
            spdlog::info("saving data to {} index", index);
        }
        return false;
    };
    std::thread store_thread;
    depth_vis.RegisterKeyCallback(GLFW_KEY_ESCAPE, callback_exit);
    color_vis.RegisterKeyCallback(GLFW_KEY_ESCAPE, callback_exit);
    pcd_vis.RegisterKeyCallback(GLFW_KEY_ESCAPE, callback_exit);

    depth_vis.RegisterKeyCallback(GLFW_KEY_SPACE, callback_save);
    color_vis.RegisterKeyCallback(GLFW_KEY_SPACE, callback_save);
    pcd_vis.RegisterKeyCallback(GLFW_KEY_SPACE, callback_save);
    bool is_geometry_added = false;
    rs.StartCapture(false);

    
    while(!flag_exit){
        auto rgbd_t = rs.CaptureFrame(true, true);
        // auto pcd_t = open3d::t::geometry::PointCloud::CreateFromRGBDImage(rgbd_t, sensor_intrinsic, open3d::core::Tensor::Eye(4, open3d::core::Float32, ((open3d::core::Device)("CPU:0"))),
        //                                                                     1000.f, 15.0f);
        // auto pcd_ = pcd_t.ToLegacy();
        // pcd = std::shared_ptr<open3d::geometry::PointCloud>(&pcd_, [](open3d::geometry::PointCloud*){});
        auto rgbd = rgbd_t.ToLegacy();

        depth_image = std::shared_ptr<open3d::geometry::Image>(&rgbd.depth_, [](open3d::geometry::Image*){});
        color_image = std::shared_ptr<open3d::geometry::Image>(&rgbd.color_, [](open3d::geometry::Image*){});

        cv::Mat temp_cv, chessboard;
        open3d::geometry::Image temp_o3d;
        std::vector<std::vector<cv::Point2f>> chessboard_points;

        temp_cv = open3d2cv(*color_image);
        target_detection(temp_cv, chessboard);
        temp_o3d = cv2open3d(chessboard);
        color_image = std::shared_ptr<open3d::geometry::Image>(&temp_o3d, [](open3d::geometry::Image*){});

        auto pcd_t = open3d::t::geometry::PointCloud::CreateFromRGBDImage(rgbd_t, sensor_intrinsic, open3d::core::Tensor::Eye(4, open3d::core::Float32, ((open3d::core::Device)("CPU:0"))),
                                                                            1000.f, 15.0f);
        auto pcd_ = pcd_t.ToLegacy();
        pcd = std::shared_ptr<open3d::geometry::PointCloud>(&pcd_, [](open3d::geometry::PointCloud*){});
 
        if(!is_geometry_added){
            if (/*!depth_vis.CreateVisualizerWindow(
                        "Open3D || RealSense || Depth", depth_image->width_,
                        depth_image->height_, 15, 50) ||
                !depth_vis.AddGeometry(depth_image) ||*/
                !color_vis.CreateVisualizerWindow(
                        "Open3D || RealSense || Color", color_image->width_,
                        color_image->height_, 675, 50) ||
                !color_vis.AddGeometry(color_image)/*||
                !pcd_vis.CreateVisualizerWindow(
                        "Open3D || RealSense || pcd", color_image->width_,
                        color_image->height_, 675, 50) ||
                !pcd_vis.AddGeometry(pcd)*/) {
                open3d::utility::LogError("Window creation failed!");
                return 0;
                }
                else{
                    flag_start = true;
                }
            is_geometry_added = true;
            // store_thread = std::thread([&](){
            //     while(!flag_exit){
            //         // spdlog::info("saving data");
            //         if(index % 10 == 0){
            //             open3d::io::WriteImage("../data/color/"+std::to_string(index)+".png", *color_image);
            //             open3d::io::WriteImage("../data/depth/"+std::to_string(index)+".png", *depth_image);
            //             open3d::io::WritePointCloudOption opt;
            //             open3d::io::WritePointCloud("../data/pcd/"+std::to_string(index)+".pcd", *pcd, opt);
            //         }
            //         index ++;
            //     }
            // });
            // if(store_thread.joinable())
            //     store_thread.join();
        }
        // open3d::io::WriteImage("../data/color/"+std::to_string(index)+".png", *color_image);
        // open3d::io::WriteImage("../data/depth/"+std::to_string(index)+".png", *depth_image);
        // open3d::io::WritePointCloudOption opt;
        // open3d::io::WritePointCloud("../data/pcd/"+std::to_string(index)+".pcd", *pcd, opt);
        // index ++;
        // depth_vis.UpdateGeometry();
        color_vis.UpdateGeometry();
        // pcd_vis.UpdateGeometry();
        // depth_vis.PollEvents();
        color_vis.PollEvents();
        // pcd_vis.PollEvents();
        // depth_vis.UpdateRender();
        color_vis.UpdateRender();
        // pcd_vis.UpdateRender();
    }
    rs.StopCapture();
    return 0;
}
