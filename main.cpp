////
//// Created by firas on 20.4.2020.
////
//
//// C++
#include <armadillo>
#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>
#include "pnp_math.h"
#include "Coresets.h"
#include "mkl.h"
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;
/**  GLOBAL VARIABLES  **/

string datapath="Data/";
/**  Functions headers  **/
void optimalpnp(std::vector<Point3d> list_points3d_inliers,std::vector<Point3d> list_lines_inliers,std::vector<double> weights,int modelindex,float *Rvect,float *tvect);
void  Load_input_data(vector<Point3d> &list_points3d_model,std::string filepath);
// can be added to the coreset class (in full project its in the frame class)
void calculate_line_matrix(std::vector<cv::Point3d> list_lines ,std::vector<cv::Mat> &list_lines_4coreset);
void fi_SVD( cv::Mat A_Mat, double *vtt);

// can be added to the coreset class (in full project its in the frame class)
cv::Mat Null_space(cv::Point3d lin);
// Converts a given integer to a string
std::string IntToString ( int Number );
void split(std::string const &str, const char delim, std::vector<std::string> &out);

/**  Main program  **/
//Accurate Coreset usage example
int main(int argc, char *argv[]) {
//    Load points
    std::vector<cv::Point3d>list_points3d;
    Load_input_data(list_points3d,"corpoints1");
//    Load lines
    std::vector<cv::Point3d> list_lines;
    Load_input_data(list_lines,"corlines1");
//    calculate null space for each direction vector(line)
    std::vector<cv::Mat> lines4coreset;
    calculate_line_matrix(list_lines ,lines4coreset);
//    initialize weights and alignment for pnp call on full data
    float rv[9];
    float tv[3];
//    weight for all pairs of (line,point) equals 1
    Mat ones= Mat::ones(list_lines.size(), 1, CV_64F);
    std::vector<double> weights=ones;
//    optimal pnp call (right now its system exe soon it will be added to the project)
    optimalpnp(list_points3d,list_lines,weights,0,rv,tv );
//    from vector to cv::Mat neeeds update (optimize)
    cv::Mat Rotation_full_data(3,3,CV_32F,rv);
    cv::Mat translation_full_data(3,1,CV_32F,tv);
//    PnP on full data output alignment R,t
    Rotation_full_data.convertTo(Rotation_full_data,CV_64F);
    translation_full_data.convertTo(translation_full_data,CV_64F);
//    calculate alignment's cost on full data
    Rotation_full_data=Rotation_full_data.t();translation_full_data=-translation_full_data;

    double full_data_cost=calculate_cost(list_points3d,list_lines,Rotation_full_data,translation_full_data);
//    initialize Coreset object
    Coresets coreset;
//    timer
    auto start1 = std::chrono::steady_clock::now();
//    ************************** Coreset Call *************************************
//    calculate accurate coreset
    std::vector<double> onese(list_points3d.size(),1.0);
    coreset.accurate_pnp_coreset(list_points3d,lines4coreset,onese);
//    ************************** Coreset Done *************************************
//end timer
    auto end1 = std::chrono::steady_clock::now();
//    time
    std::chrono::duration<double> elapsed_seconds1 = end1-start1;
//    get Coreset output , weights & indexes
    std::vector<int> Coreset_Indexes= coreset.get_coreset_indexes();
    std::vector<double> Coreset_w= coreset.get_coreset_weights();
//    w=w^2
//    transform(Coreset_w.begin(), Coreset_w.end(), Coreset_w.begin(), [](double x){return x*x;});
//    select coreset lines and points
    std::vector<cv::Point3d>Coreset_points3d;
    std::vector<cv::Point3d> Coreset_lines;
    for (int k = 0; k <Coreset_Indexes.size() ; ++k) {
        Coreset_points3d.push_back(list_points3d[Coreset_Indexes[k]]);
        Coreset_lines.push_back(list_lines[Coreset_Indexes[k]]);
    }
//    initialize alignment for pnp call on coreset data
    float rv2[9];
    float tv2[3];
//    optimal pnp call on coreset (right now its system exe soon it will be added to the project)
    optimalpnp(Coreset_points3d,Coreset_lines,Coreset_w,1,rv2,tv2);
//    PnP on coreset data output alignment R,t
    cv::Mat Rotation_coreset(3,3,CV_32F,rv2);
    cv::Mat translation_coreset(3,1,CV_32F,tv2);
    Rotation_coreset.convertTo(Rotation_coreset,CV_64F);
    translation_coreset.convertTo(translation_coreset,CV_64F);
    Rotation_coreset=Rotation_coreset.t();translation_coreset=-translation_coreset;
//    calculate alignment's cost on Coreset

    double Coreset_cost=calculate_cost(list_points3d,list_lines,Rotation_coreset,translation_coreset);
    double weighted_coreset_cost=calculate_weighted_cost(Coreset_points3d,Coreset_lines,Rotation_coreset,translation_coreset,Coreset_w);
    cout.precision(60);
    cout<<"full data cost = "<<full_data_cost<<endl;
    cout<<"Coreset cost = "<<Coreset_cost<<endl;
    cout<<"weighted coreset cost = "<<weighted_coreset_cost<<endl;
    cout<<"Cost Ratio :\n full_data_cost/Coreset_cost  = "<< full_data_cost/Coreset_cost<<endl;
    cout<<"create coreset Time:"<<elapsed_seconds1.count()<<endl;
    cout<<Rotation_coreset<<endl;
    cout<<Rotation_full_data<<endl;
    cout<<translation_coreset<<endl;
    cout<<translation_full_data<<endl;

}
cv::Mat Null_space(cv::Point3d lin){
    cv::Mat  A(lin),w, vt;

    auto start1 = std::chrono::steady_clock::now();
#define M 1
#define N 3
#define LDA M
#define LDU M
#define LDVT N
    double vtt[LDVT*N];
    fi_SVD(A,vtt);
//    cv::Mat At=A.t();
//    arma::mat arma_A( reinterpret_cast<double*>(At.data), A.rows, A.cols );
//    arma::mat U, Var;
//    arma::vec S;
//    arma::svd(U, S, Var, arma_A,"dc");
//    cv::Mat u(  A.rows,A.rows, CV_64F, U.memptr() );
    cv::Mat u(  A.rows,A.rows, CV_64F, vtt );
    auto end1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = end1-start1;
//    cout<<elapsed_seconds1.count()<<endl;
    cv::Mat uu;
    uu.push_back(u.row(1));
    uu.push_back(u.row(2));
    uu.push_back(u.row(0));
//    cout<<uu<<endl;
    return uu;
}
//[0.2915089212452594, 0.9163643503331068, 0.2744064982339308;
//-0.9564335615429169, 0.2744064982339307, 0.09967906541156468;
//0.01604341574467139, -0.2915089212452594, 0.9564335615429169]
void calculate_line_matrix(std::vector<cv::Point3d> list_lines ,vector<cv::Mat> &list_lines_4coreset){
    for(auto line: list_lines){
        cv::Mat lineMatrix;
//        calculate null space of diretion vector
        cv::Mat templinematrix= Null_space(line);
        lineMatrix=templinematrix(cv::Range(0, templinematrix.rows - 1), cv::Range(0, templinematrix.cols)).clone();
        list_lines_4coreset.push_back(lineMatrix);
    }
}

void optimalpnp(std::vector<Point3d> list_points3d_inliers,std::vector<Point3d> list_lines_inliers,std::vector<double> weights,int modelindex,float *Rvect,float *tvect)
{
//    MUST BE CHANGED ASAP
//  initialize input txt file for the optimal pnp exe
    int numofinliers=list_points3d_inliers.size();
    string matstring="Mattest"+IntToString(modelindex)+".txt";
    ofstream matfile;
    matfile.open (matstring);
//    first line in the folder = number of points , number of input variables=10 ([v1,v2,v3,c1,c2,c3,x1,x2,x3,w]) size of T3x10 (30)
    matfile << IntToString(numofinliers)<<" 10 30\n";
//    rest of the file is the data pair(xi,vi)=[vi1,vi2,vi3,ci1,ci2,ci3,xi1,xi2,xi3,wi], note: c= (0,0,0) always as its the center of the camera for all lines
    for (int i = 0; i < numofinliers; i++)
    {
        matfile << std::fixed << std::setprecision(30)<<format("%.64f %.64f %.64f ", list_lines_inliers[i].x, list_lines_inliers[i].y,list_lines_inliers[i].z)<<" "<<"0 0 0 "<<
                format("%.64f %.64f %.64f ", list_points3d_inliers[i].x, list_points3d_inliers[i].y,list_points3d_inliers[i].z)<<" "<<double(weights[i])<<"\n";
    }
    matfile << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n";
    matfile.close();
//    system call for the optimal pnp
    int ret=system(("echo firasa92205 | sudo -S ./pnp_firs "+ matstring+" OPnP_R_"+IntToString(modelindex)+".txt "+ "OPnP_t_"+IntToString(modelindex)+".txt >/dev/null").c_str());
//    read results
    string line;
    const char delim = ',';
    int row=0;
    ifstream rotfile ("OPnP_R_"+IntToString(modelindex)+".txt");
    ifstream transfile ("OPnP_t_"+IntToString(modelindex)+".txt");
    int count=0;
//    no need to waste time on this it works right , and it will be changed soon anyways
    if (rotfile.is_open()){
        while ( getline (rotfile,line) )
        {
            std::vector<std::string> out;
            split(line, delim, out);
            int col =0 ;
            for (auto &s: out) {
                double temp = (double)atof(s.c_str());
                Rvect[count]=temp;
                count+=1;
                col+=1;
            }
            row+=1;
        }
        rotfile.close();
    }
    row=0;
    if (transfile.is_open()){
        while ( getline (transfile,line) )
        {
            std::vector<std::string> out;
            split(line, delim, out);
            for (auto &s: out) {
                double temp = (double)atof(s.c_str());
                tvect[row]=temp;
                row+=1;
            }
        }
        transfile.close();
    }
}

void  Load_input_data( vector<Point3d> &list_points3d_model,std::string filepath){
// worst way to read input data.
    int counter = 0;
    string x;
    string y;
    string z;
    ifstream point3dfile;
    point3dfile.open(filepath+".txt");
    int i = 0;
    while (point3dfile.good()) {
        if (counter == 0)
        {
            getline(point3dfile, x, ' ');
            counter++;
        }
        if (counter == 1)
        {
            getline(point3dfile, y, ' ');
            counter++;
        }
        if (counter == 2)
        {
            getline(point3dfile, z, '\n');
            if(x==""){
                int a=0;
                break;}
            double xF = ::atof(x.c_str());
            double yF = ::atof(y.c_str());
            double zF = ::atof(z.c_str());
            list_points3d_model.push_back({ xF, yF, zF });
            string x;
            string y;
            string z;
            counter = 0;
            i++;
        }
    }
}
// Converts a given integer to a string
std::string IntToString ( int Number )
{
    std::ostringstream ss;
    ss << Number;
    return ss.str();
}

void split(std::string const &str, const char delim,
           std::vector<std::string> &out)
{
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

void fi_SVD(cv::Mat A_Mat, double *vt){
    std::vector<double> a_vector=A_Mat;
    double *a=&a_vector[0];

    MKL_INT m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
    double wkopt;
    double* work;
    /* iwork dimension should be at least 8*min(m,n) */
    MKL_INT iwork[8*N];
    double s[N], u[LDU*M];
    /* Executable statements */
//    printf( " DGESDD Example Program Results\n" );
    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgesdd( "A", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt,
            &lwork, iwork, &info );
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    /* Compute SVD */
//    auto start2 = std::chrono::steady_clock::now();
    dgesdd( "A", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
            &lwork, iwork, &info );
//    auto end2 = std::chrono::steady_clock::now();
//    std::chrono::duration<double> elapsed_seconds2 = end2-start2;
//    std::cout<<"kakaka"<<elapsed_seconds2.count()<<endl;
//    print_matrix( "Singular values", 1, n, s, 1 );
//    /* Print left singular vectors */
//    print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
//    /* Print right singular vectors */
//    print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
//    cv::Mat A(nin, nin, CV_64F,vt);
//    A.copyTo(newMat);
//    cout<<A<<endl;
//    int bobo=0;
}