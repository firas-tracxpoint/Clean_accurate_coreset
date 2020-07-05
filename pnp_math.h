
// Created by firas on 9.4.2020.



// OpenCV
#include <opencv2/core.hpp>


//this class is used only for pnp cost calculation for now
#ifndef UNTITLED_PNP_MATH_H
#define UNTITLED_PNP_MATH_H
//using namespace cv;
//using namespace std;
double calculate_cost(std::vector<cv::Point3d> &points,std::vector<cv::Point3d> &lines,cv::Mat &Rotation , cv::Mat &translation);
double calculate_weighted_cost(std::vector<cv::Point3d> &points,std::vector<cv::Point3d> &lines,cv::Mat &Rotation , cv::Mat &translation,std::vector<double> core_w);
#endif //UNTITLED_PNP_MATH_H
