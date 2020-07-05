//
// Created by firas on 9.4.2020.
//


#include <bits/stdc++.h>

#include "pnp_math.h"
cv::Mat I2= cv::Mat::eye(3, 3, CV_32FC1);
cv::Mat I3= cv::Mat::eye(3, 3, CV_64F);
//using namespace cv;
//using namespace std;
/**
 * @summary calculates the cost of the PnP output alignment cost=sum(||(R*p-t)*(I-vv')||^2)
 *
 * @points std::vector<Point3f>& 3d points Float
 * @lines std::vector<Point3f>& 3d points represinting direction vector of pixels Float
 * @Rotation cv::Mat& Rotation matrix
 * @translation cv::Mat& translation vector
 * @return cost=sum(||(R*p-t)*(I-vv')||^2)
 */
float calculate_cost(std::vector<cv::Point3f> &points,std::vector<cv::Point3f> &lines,cv::Mat &Rotation , cv::Mat &translation){
    float sumcost=0;
    int n=points.size();
    for (int indx = 0; indx < n ; indx++){
        cv::Mat ll=cv::Mat(lines[indx]);
        cv::Mat vtv = cv::Mat(lines[indx])*cv::Mat(lines[indx]).t();
        cv::Mat pp=cv::Mat(points[indx]);
        cv::Mat Rpminust=((Rotation*cv::Mat(points[indx]))-translation).t()*(I2-vtv);
        float cost=norm(Rpminust)*norm(Rpminust);
        sumcost+=(cost/n);
    }
    return sumcost;
}


/**
 * @summary calculates the cost of the PnP output alignment cost=sum(||(R*p-t)*(I-vv')||^2)
 *
 * @points std::vector<Point3f>& 3d points Double
 * @lines std::vector<Point3f>& 3d points represinting direction vector of pixels Double
 * @Rotation cv::Mat& Rotation matrix
 * @translation cv::Mat& translation vector
 * @return cost=sum(||(R*p-t)*(I-vv')||^2)
 */
double calculate_cost(std::vector<cv::Point3d> &points,std::vector<cv::Point3d> &lines,cv::Mat &Rotation , cv::Mat &translation){
    double sumcost=0;
    int n=points.size();
    for (int indx = 0; indx < n ; indx++){
        cv::Mat vtv = cv::Mat(lines[indx])*cv::Mat(lines[indx]).t();
        cv::Mat Rpminust=((Rotation*cv::Mat(points[indx]))-translation).t()*(I3-vtv);
        double cost=norm(Rpminust)*norm(Rpminust);
        sumcost+=(cost/n);
    }
    return sumcost;
}

double calculate_weighted_cost(std::vector<cv::Point3d> &points,std::vector<cv::Point3d> &lines,cv::Mat &Rotation , cv::Mat &translation,std::vector<double> core_w){
    double sumcost=0;
    int n=points.size();
    double sumw= std::accumulate(core_w.begin(), core_w.end(), 0.0);
    int sumsum=0;
    for (int indx = 0; indx < n ; indx++){
        cv::Mat vtv = cv::Mat(lines[indx])*cv::Mat(lines[indx]).t();
        cv::Mat Rpminust=((Rotation*cv::Mat(points[indx]))-translation).t()*(I3-vtv);
        double cost=norm(Rpminust)*norm(Rpminust);
        sumcost+=(cost*(core_w[indx]/sumw));
    }
    return sumcost;
}