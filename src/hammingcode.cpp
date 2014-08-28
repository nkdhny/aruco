#include "hammingcode.h"

namespace nkdhny{

int HammingCode::rotate(const cv::Mat &in, cv::Mat& out)
{
    cv::Mat rotation(5,5, CV_8UC1);
    cv::Mat prevRotation(5,5, CV_8UC1);
    cv::Mat bestRotation(5,5, CV_8UC1);

    int dist;
    int minimumDistance;
    int rotationsMade = 0;

    in.copyTo(rotation);
    rotation.copyTo(bestRotation);

    dist = distance(rotation);
    minimumDistance = dist;

    for (int i=1;i<4;i++)
    {
        rotation.copyTo(prevRotation);
        apply_rotation(prevRotation, rotation);
        //get the hamming distance to the nearest possible word
        dist=distance(rotation);
        if (dist < minimumDistance)
        {
            rotationsMade = i;
            minimumDistance = dist;
            rotation.copyTo(bestRotation);
        }
    }

    if(minimumDistance == 0){
        bestRotation.copyTo(out);
        return rotationsMade;
    } else {
        return -1;
    }
}

int HammingCode::apply_rotation(const cv::Mat &in, cv::Mat &out)
{
    in.copyTo(out);
    for (int i=0;i<in.rows;i++)
    {
        for (int j=0;j<in.cols;j++)
        {
            out.at<uchar>(i,j)=in.at<uchar>(in.cols-j-1,i);
        }
    }
    return 0;
}

int HammingCode::distance(const cv::Mat &bits)
{
    int ids[4][5]=
    {
        {
            1, 0, 0, 0, 0
        }
        ,
        {
            1, 0, 1, 1, 1
        }
        ,
        {
            0, 1, 0, 0, 1
        }
        ,
        {
            0, 1, 1, 1, 0
        }
    };
    int dist=0;

    for (int y=0;y<5;y++)
    {
        int minSum=1e5;
        //hamming distance to each possible word
        for (int p=0;p<4;p++)
        {
            int sum=0;
            //now, count
            for (int x=0;x<5;x++)
                sum+=  bits.at<uchar>(y,x) == ids[p][x]?0:1;
            if (minSum>sum) minSum=sum;
        }
        //do the and
        dist+=minSum;
    }

    return dist;
}

int HammingCode::decode(const cv::Mat &in)
{
    int matID=0;
    cv::Mat bits(5,5,CV_8UC1);
    in.copyTo(bits);
    rotate(in, bits);
    for (int y=0;y<5;y++)
    {
        matID<<=1;
        if ( bits.at<uchar>(y,1)) matID|=1;
        matID<<=1;
        if ( bits.at<uchar>(y,3)) matID|=1;
    }
    return matID;
}

int HammingCode::encode(int id, cv::Mat &out)
{
    cv::Mat marker(5,5, CV_8UC1);
    marker.setTo(cv::Scalar(0));
    if (0<=id && id<1024) {
        //for each line, create
        int ids[4]={0x10,0x17,0x09,0x0e};
        for (int y=0;y<5;y++) {
            int index=(id>>2*(4-y)) & 0x0003;
            int val=ids[index];
            for (int x=0;x<5;x++) {
                if ( ( val>>(4-x) ) & 0x0001 ) marker.at<uchar>(y,x)=1;
                else marker.at<uchar>(y,x)=0;
            }
        }
    }
    else {
        return -1;
    }
    marker.copyTo(out);
    return 0;
}

}
