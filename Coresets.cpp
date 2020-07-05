//
// Created by firas on 12.4.2020.
//

#include <armadillo>
#include <cxcore.h>
#include "Coresets.h"
#include <tuple>
#include <vector>
#include <random>
#include <algorithm>
#include "pnp_math.h"
#include "Coresets.h"
#include <chrono>
#include <stdlib.h>
#include "mkl.h"

const int d=3;
Coresets::Coresets() : coreset_weights_(0), coreset_indexes_(0)
{
}
Coresets::~Coresets()
{
    // TODO Auto-generated destructor stub
}

bool isLargerthanZero(double x){
    return x > 0;
}

//using MKL (intel math kernel library) for svd !!
void mkl_SVD(cv::Mat A_Mat, double *u){

    std::vector<double> a_vector;
    if (A_Mat.isContinuous()) {
        a_vector.assign((double*)A_Mat.data, (double*)A_Mat.data + A_Mat.total());
    } else {
        for (int i = 0; i < A_Mat.rows; ++i) {
            a_vector.insert(a_vector.end(), A_Mat.ptr<double>(i), A_Mat.ptr<double>(i)+A_Mat.cols);
        }
    }
    double *a=&a_vector[0];
    MKL_INT m = A_Mat.cols, n = A_Mat.rows, lda = A_Mat.cols, ldu = A_Mat.cols, ldvt = A_Mat.rows, info, lwork;
    double wkopt;
    double* work;
    /* iwork dimension should be at least 8*min(m,n) */
    MKL_INT iwork[8*n];
    double s[n], vt[n*n];
    /* Executable statements */
    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgesdd( "A", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt,
            &lwork, iwork, &info );
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    /* Compute SVD */
    dgesdd( "A", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
            &lwork, iwork, &info );
}
/**
 * @summary cara theodory algorithm
 *
 * @P Matrix of size nxd containing n points of d dimensions
 * @u cv::Mat weights of size nx1
 * @finalP cv::Mat& output points
 * @finalu cv::Mat& output weights
 * @return void function that updates finalP&finalu
 */
void CaraIdxCoreset(cv::Mat P, cv::Mat u,cv::Mat &finalP ,cv::Mat &finalu) {
    double minval,maxval;
    cv::Point p1,p2;
    while (1) {
        int n = cv::countNonZero(u);
        int d = P.cols;
        if(n<=d+1) {
//          coreset done
            finalP = P.clone();
            finalu = u.clone();
            return;
        }
//start python line:  u_non_zero = np.nonzero(u) MUST be a better way to do this!
//fin where weights equals zero
        cv::Mat u_non_zero2, u_non_zero;
        cv::inRange(u, cv::Scalar(std::numeric_limits<double>::min()),
                    cv::Scalar(std::numeric_limits<double>::infinity()), u_non_zero2);
        cv::findNonZero(u_non_zero2, u_non_zero);
//end python line:  u_non_zero = np.nonzero(u)
//python  A = P[u_non_zero];
//chose the points that has weights larger than 0
        cv::Mat A;
        for (int i = 0; i < u_non_zero.rows; ++i) {
            A.push_back(P.row(u_non_zero.at<int>(i, 1)));
        }
        cv::Mat ones(A.rows - 1, 1, CV_64F, cv::Scalar(1));
        cv::Mat reduced_vec = A.row(0).t() * ones.t();
//python  A = A[1:].T - reduced_vec
        A = A(cv::Range(1, A.rows), cv::Range::all()).t() - reduced_vec;

// initialize empty array for mkl svd
        double vtt[A.cols*A.cols];
        mkl_SVD(A,vtt);
//  data manipulation (from std array to cv::Mat) once we stop using opencv remove this
        cv::Mat V(  A.cols,A.cols, CV_64F, vtt );
//  chose the minimal eigen vector
        cv::Mat v = V.row(V.rows - 1).t();
        double constant = 0.000001;
//        cout<<v<<endl;
//python line : diff = np.max(np.abs(np.dot(A, v )))
        cv::minMaxLoc(cv::abs(A*v), &minval, &maxval, &p1, &p2);
        double diff = maxval;
        /*while  (diff) > const and idx_of_try < A.shape[1]  :
        try_matrix = np.delete(A, idx_of_try, axis=1)
        try_b  =  -1*  A[:,idx_of_try]
        v= lstsq(try_matrix, try_b,cond=cond)[0] ;
        diff = np.max(np.abs(np.dot(try_matrix, v )  - try_b))
        if diff > const*/
        if (diff > constant) {
            return;
        }
        std::vector<double> v_array=v;
        v_array.insert(v_array.begin(), -1*cv::sum(v)[0]);
//start python line: idx_good_alpha = np.nonzero(v > 0)
        cv::Mat idx_good_alpha;
        std::vector<double>::iterator iter = v_array.begin();
        while ((iter = std::find_if(iter, v_array.end(), isLargerthanZero)) != v_array.end()) {
            int idx = std::distance(v_array.begin(), iter);
            idx_good_alpha.push_back(idx);
            iter++;
        }
//end python line: idx_good_alpha = np.nonzero(v > 0)
        cv::Mat u_where_nonZero,matforalpha;
        for (int id = 0; id < u_non_zero.rows; id++) {
            u_where_nonZero.push_back(u.at<double>(u_non_zero.at<int>(id, 1)));}
//start python line: alpha = np.min(u[u_non_zero][idx_good_alpha]/v[idx_good_alpha])
        for (int j = 0; j < idx_good_alpha.rows; j++) {
                matforalpha.push_back(u.at<double>(u_non_zero.at<int>(idx_good_alpha.at<int>(j), 1))/v_array[idx_good_alpha.at<int>(j)]);
            }
        cv::minMaxLoc(matforalpha,&minval,&maxval,&p1,&p2);
        double alpha=minval;
//end python line: alpha = np.min(u[u_non_zero][idx_good_alpha]/v[idx_good_alpha])
        cv::Mat w=cv::Mat::zeros(cv::Size(1,u.rows), CV_64F);
        cv::Mat temp=u_where_nonZero-(alpha*cv::Mat(v_array));
        cv::minMaxLoc(temp,&minval,&maxval,&p1,&p2);
        temp.at<double>(p1)=double(0.0);
//python line :   w[u_non_zero] = tmp
        for (int l = 0; l < u_non_zero.rows; l++) {
            w.at<double>(u_non_zero.at<int>(l,1))=temp.at<double>(l);
        }
        u=w.clone();
    }

}
/**
 * @summary this function is used to further optimize the cara theodry algorithm in function CaraIdxCoreset
 *
 * @P Matrix of size nxd containing n points of d dimensions
 * @w weights
 * @coreset_size integer of the coreset's size
 * @return std::make_tuple(points,weights,indexes) of the coreset
 */
std::tuple<cv::Mat,cv::Mat,cv::Mat> update_cara(cv::Mat P,cv::Mat w, int coreset_size){

    int d=P.cols ; int n=P.rows; int m=2*d+2 ; //int n=P.rows
    if (n<=d) {
        cv::Mat idx;
        return std::make_tuple(P, w,idx);
    }
    double w_sum=cv::sum( w)[0];
    w=w/w_sum;
    int chunk_size=ceil(double(n)/double(m));
    int current_m=ceil(double(n)/double(chunk_size));
    int add_z=chunk_size-int(n%chunk_size);
    if (add_z!=chunk_size){
        cv::Mat zeros=cv::Mat::zeros(cv::Size(P.cols,add_z), CV_64F);
        cv::vconcat(P,zeros,P);
        zeros=cv::Mat::zeros(cv::Size(w.cols,add_z), CV_64F);
        cv::vconcat(w,zeros,w);
    }
    std::vector<int> idx_group(P.rows);

    std::iota( std::begin( idx_group ), std::end( idx_group ), 0 );

    std::vector<cv::Mat> p_groups;
    cv::Mat temp_P;
    temp_P=P.clone();
    for (int l = 0; l <current_m ; l++) {
        cv::Mat slice(chunk_size, P.cols, CV_64F, temp_P.ptr<double>(chunk_size*(l),0)); //dim,rows,col
        p_groups.push_back(slice.clone());
    }
    cv::Mat w_groups=w.reshape(0,current_m);
    int w_nonzero;
    w_nonzero= cv::countNonZero(idx_group);
    if (not coreset_size) {
        coreset_size = d + 1;
    }
    cv::Mat final_P,final_w,final_idx_array;
    while (w_nonzero>coreset_size){
//      start Python line groups_means = np.einsum('ijk,ij->ik',p_groups, w_groups)
//      needs optimizing !!!!!!
        std::vector<cv::Mat> lala =std::vector<cv::Mat>(p_groups.begin() , p_groups.begin() +current_m);
        cv::Mat groups_means=cv::Mat::zeros(cv::Size(p_groups[0].cols,w_groups.rows), CV_64F);
        cv::Mat new_P,new_w,new_idx_array;
        for (int i = 0; i <chunk_size ; ++i) {
            cv::Mat p_chunk;
            cv::Mat w_chunk_col2=w_groups.col(i);
            cv::Mat w_chunk2=w_chunk_col2;
            cv::Mat temc;
            int chunkcount=0;
            for(auto chunk: lala){
                temc=chunk.row(i).mul((w_chunk2.row(chunkcount)));
                p_chunk.push_back(temc);
                chunkcount++;
            }
            groups_means+=p_chunk;

        }
//      end Python line groups_means = np.einsum('ijk,ij->ik',p_groups, w_groups)

//      python line  group_weigts = np.ones(groups_means.shape[0], dtype = dtype)*1/current_m
        cv::Mat groups_weights( groups_means.rows,1,CV_64F,cv::Scalar(double(1.0/current_m)) );
        cv::Mat Cara_P,Cara_w_idx;
//        Call Cara algorithm
        CaraIdxCoreset(groups_means,groups_weights,Cara_P,Cara_w_idx);

        if(Cara_P.empty()) {
//            Failed
            cv::Mat none;
            return std::make_tuple(Cara_P, Cara_w_idx, none);
        }
//       start python line :         IDX = np.nonzero(Cara_w_idx)
        cv::Mat IDX, cara_non_zero;
        cv::inRange(Cara_w_idx, cv::Scalar(std::numeric_limits<double>::min()),
                    cv::Scalar(std::numeric_limits<double>::infinity()), cara_non_zero);
        cv::findNonZero(cara_non_zero, IDX);
//       end python line :         IDX = np.nonzero(Cara_w_idx)

//      start python lines  new_P =
//      1- p_groups[IDX].reshape(-1,d)
//      2- new_w = (current_m * w_groups[IDX] * Cara_w_idx[IDX][:, np.newaxis]).reshape(-1, 1)
//      3- new_idx_array = idx_group[IDX].reshape(-1,1)
        for (int k = 0; k <IDX.rows ; ++k) {
            int IDX_ind=IDX.at<int>(k,1);
            new_P.push_back(p_groups[IDX_ind].clone());
            std::vector<int> curr_idx(idx_group.begin()+IDX_ind*chunk_size,idx_group.begin()+IDX_ind*chunk_size+chunk_size);
            new_w.push_back((current_m*w_groups.row(IDX_ind)).t()*Cara_w_idx.row(IDX_ind));
            new_idx_array.push_back(curr_idx);
        }
//        end python lines

        w_nonzero=cv::countNonZero(new_w);
        chunk_size=ceil(double(new_P.rows)/double(m));
        current_m=ceil(double(new_P.rows)/double(chunk_size));
        add_z= chunk_size-int(new_P.rows%chunk_size);
        if (add_z!=chunk_size){
            cv::Mat zeros=cv::Mat::zeros(cv::Size(new_P.cols,add_z), CV_64F);
            cv::vconcat(new_P,zeros,new_P);
            zeros=cv::Mat::zeros(cv::Size(new_w.cols,add_z), CV_64F);
            cv::vconcat(new_w,zeros,new_w);
            std::vector<int> a={0};
            new_idx_array.push_back(a);
        }
//        start python line :
//        1- p_groups = new_P.reshape(current_m, chunk_size, new_P.shape[1])
//        2- w_groups = new_w.reshape(current_m, chunk_size)
//        3- idx_group = new_idx_array.reshape(current_m , chunk_size)
        p_groups.clear();
        cv::Mat cur_P;
        cur_P=new_P.clone();
        for (int l = 0; l <current_m ; l++) {
            cv::Mat slice(chunk_size, new_P.cols, CV_64F, cur_P.ptr<double>(chunk_size*(l),0)); //dim,rows,col
            p_groups.push_back(slice.clone());
        }
        w_groups.release();
        cv::Mat ww(current_m,chunk_size, CV_64F, new_w.data);
        w_groups=ww.clone();
        idx_group=new_idx_array;
        final_P=new_P.clone();final_w=new_w.clone(),final_idx_array=new_idx_array.clone();

    }
    return std::make_tuple(final_P,w_sum*final_w,final_idx_array);
}

/**
 * @summary creates accurate Coreset for the PnP problem
 *
 * @points3d a set of 3d points
 * @lines a set of cv::Mats of size 2x3 points (the null space of the line)
 * @return a set of indexes for the pairs of points and lines chosen to be in the coreset and the coresponding weight
 */
void Coresets::accurate_pnp_coreset(std::vector<cv::Point3d> &points3d,std::vector<cv::Mat> &lines,std::vector<double> oldweights){
    int n=points3d.size();
    cv::Mat S=generateData(points3d,lines);
    int d_new=S.rows;

// Need to take every point of the (d-1) points in S_i, turn it to a (cov) matrix, then compute the mean of the (d-1) matrices.
// The vector form of those mean-matrices is the input to cara
    cv::Mat cov=cv::Mat::zeros(cv::Size(d_new,d_new), CV_64F);
    cv::Mat S_means=cv::Mat::zeros(cv::Size(n,pow(d_new,2)), CV_64F);
    int counter=0; int curr_mean_idx=0;
    int s_cols=S.cols;
    for(int i=0;i<s_cols;i++) {
        cov = cov + (S.col(i) * S.col(i).t());
        counter++;
        // Whenever we have summed (d-1) vectors so far, turn cov into a vector, divide by d-1, and append...
        if (counter == (d - 1)) {
            S_means.col(curr_mean_idx) = cov.reshape(1, 1).t() / (d - 1);
            cov = cv::Mat::zeros(cv::Size(d_new, d_new), CV_64F);
            counter = 0;
            curr_mean_idx++;
        }
    }
    cv::Mat Ones( S_means.cols,1,CV_64F,cv::Scalar(1) );
    cv::Mat mat_oldweights=cv::Mat(oldweights);
    cv::Mat W_cara;
    cv::Mat P_cara;
    cv::Mat C_idx;
    std::tie(P_cara,W_cara,C_idx)=update_cara(S_means.t(),mat_oldweights,S_means.rows+1);

//################## Test the coreset ##############################
//    cv::Mat C_means(S_means.rows,C_idx.rows,CV_64F);
//    for (int i = 0; i < C_idx.rows; ++i)  {
//        if(C_idx.at<int>(i,0)<S_means.cols)
//            S_means.col(C_idx.at<int>(i,0)).copyTo(C_means.col(i));
//    }
//
//    cv::Mat check_corset;
//    for (int j = 0; j <C_means.cols ; ++j) {
//        cv::Mat row_j=(C_means.col(j)*W_cara.at<double>(j,0)).t();
//        check_corset.push_back(row_j);
//    }
//    cv::Mat sum_C;
//    cv::reduce(check_corset, sum_C,0,CV_REDUCE_SUM,CV_64F );
//    cv::Mat sum_S;
//    cv::reduce(S_means.t(), sum_S,0,CV_REDUCE_SUM,CV_64F );
//    if (norm(sum_S-sum_C)>0.01)
//        std::cout << "Problem with coreset" << std::endl;
//        std::cout << norm(sum_S-sum_C) << std::endl;
    coreset_weights_=W_cara;

    coreset_indexes_=C_idx;
}

/**
 * @summary calculates matrix S for the coreset
 *
 * @P std::vector<cv::Point3d> 3d points
 * @V std::vector<cv::Mat> lines
 * @return Matrix S , for ||Sx||
 */
cv::Mat Coresets::generateData(std::vector<cv::Point3d> &P,std::vector<cv::Mat> &V) {
    int n=P.size();
    cv::Mat b_curr=cv::Mat::zeros(cv::Size(1,2), CV_64F);
    cv::Mat S = cv::Mat::zeros(cv::Size(n*(d-1),(pow(d,2))+d+1), CV_64F);
// Compute s_1,...,s_{d-1} (s_0,...,s_{d-2}) for every input pair

    int counter=0;
    for (int i = 0; i <n ; ++i) {
        cv::Point3d p_curr=P[i];
        cv::Mat Vi=V[i];
        cv::Mat v_curr2;
        cv::Mat v_curr=V[i];
        cv::hconcat(v_curr,b_curr,v_curr2);
        for (int k = 0; k < d-1 ; k++) {
// Compute the vector s_k = (V[k,0]*p,...,V[k,d-1]*p,V[k,:],b[k])
            cv::Mat outerproduct_VPt=v_curr.row(k).t()*cv::Mat(p_curr).t();
            outerproduct_VPt=outerproduct_VPt.reshape(1,1);
            cv::Mat s_k;
            cv::hconcat(outerproduct_VPt,v_curr2.row(k),s_k);
            S.col(counter)=s_k.t();
            counter++;
        }
    }
    return S;
}
