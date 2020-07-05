//
// Created by firas on 12.4.2020.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifndef UNTITLED_CORESETS_H
#define UNTITLED_CORESETS_H


class Coresets {
public:
    Coresets();
    virtual ~Coresets();
    std::vector<double> get_coreset_weights(){return coreset_weights_;}
    std::vector<int> get_coreset_indexes(){return coreset_indexes_;}
    void accurate_pnp_coreset(std::vector<cv::Point3d> &points3d,std::vector<cv::Mat> &lines,std::vector<double>oldweights);
    cv::Mat generateData(std::vector<cv::Point3d> &P,std::vector<cv::Mat> &V);
    void mkl_SVD( cv::Mat A_Mat, double *vtt);

private:
    std::vector<double> coreset_weights_;
    std::vector<int> coreset_indexes_;

};


#endif //UNTITLED_CORESETS_H
