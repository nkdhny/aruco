/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/

#ifndef ArucoFiducicalMarkerDetector_H
#define ArucoFiducicalMarkerDetector_H
#include <opencv2/core/core.hpp>
#include "exports.h"
#include "marker.h"
#include "board.h"

#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace aruco {

template <typename CodeType> class ARUCO_EXPORTS FiducidalMarkers {
public:
    /**
    * \brief Creates an ar marker with the id specified using a modified version of the hamming code.
    * There are two type of markers: a) These of 10 bits b) these of 3 bits. The latter are employed for applications
    * that need few marker but they must be small.  The two type of markers are distinguished by their ids. While the first type
    * of markers have ids in the interval [0-1023], the second type ids in the interval [2000-2006].
    *
    *
    * 10 bits markers
    * -----------------------
    * There are a total of 5 rows of 5 cols. Each row encodes a total of 2 bits, so there are 2^10 bits:(0-1023).
    *
    * The least significative bytes are first (from left-up to to right-bottom)
    *
    * Example: the id = 110 (decimal) is be represented in binary as : 00 01 10 11 10.
    *
    * Then, it will generate the following marker:
    *
    * -# 1st row encodes 00: 1 0 0 0 0 : hex 0x10
    * -# 2nd row encodes 01: 1 0 1 1 1 : hex 0x17
    * -# 3nd row encodes 10: 0 1 0 0 1 : hex 0x09
    * -# 4th row encodes 11: 0 1 1 1 0 : hex 0x0e
    * -# 5th row encodes 10: 0 1 0 0 1 : hex 0x09
    *
    * Note that : The first bit, is the inverse of the hamming parity. This avoids the 0 0 0 0 0 to be valid
    * These marker are detected by the function  getFiduciadlMarker_Aruco_Type1
    * @param writeIdWaterMark if true, writes a watermark with the marker id
    */
    static cv::Mat createMarkerImage(int id,int size,bool writeIdWaterMark=true) throw (cv::Exception);

    /** Detection of fiducidal aruco markers (10 bits)
     * @param in input image with the patch that contains the possible marker
     * @param nRotations number of 90deg rotations in clockwise direction needed to set the marker in correct position
     * @return -1 if the image passed is a not a valid marker, and its id in case it really is a marker
     */
    static int detect(const cv::Mat &in,int &nRotations);

    /**Similar to createMarkerImage. Instead of returning a visible image, returns a 8UC1 matrix of 0s and 1s with the marker info
     */
    static cv::Mat getMarkerMat(int id) throw (cv::Exception);


    /**Creates a printable image of a board
     * @param gridSize grid layout (numer of sqaures in x and Y)
     * @param MarkerSize size of markers sides in pixels
     * @param MarkerDistance distance between the markers
      * @param TInfo output 
     * @param excludedIds set of ids excluded from the board
     */
    static  cv::Mat createBoardImage( cv::Size  gridSize,int MarkerSize,int MarkerDistance,  BoardConfiguration& TInfo ,vector<int> *excludedIds=NULL ) throw (cv::Exception);


    /**Creates a printable image of a board in chessboard_like manner
     * @param gridSize grid layout (numer of sqaures in x and Y)
     * @param MarkerSize size of markers sides in pixels
      * @param TInfo output 
     * @param setDataCentered indicates if the center is set at the center of the board. Otherwise it is the left-upper corner
     * 
     */
    static  cv::Mat  createBoardImage_ChessBoard( cv::Size gridSize,int MarkerSize, BoardConfiguration& TInfo ,bool setDataCentered=true ,vector<int> *excludedIds=NULL) throw (cv::Exception);

    /**Creates a printable image of a board in a frame fashion 
     * @param gridSize grid layout (numer of sqaures in x and Y)
     * @param MarkerSize size of markers sides in pixels
     * @param MarkerDistance distance between the markers
      * @param TInfo output 
     * @param setDataCentered indicates if the center is set at the center of the board. Otherwise it is the left-upper corner
     * 
     */
    static  cv::Mat  createBoardImage_Frame( cv::Size gridSize,int MarkerSize,int MarkerDistance,  BoardConfiguration& TInfo ,bool setDataCentered=true,vector<int> *excludedIds=NULL ) throw (cv::Exception);

private:
  
    static vector<int> getListOfValidMarkersIds_random(int nMarkers,vector<int> *excluded) throw (cv::Exception);
    static int analyzeMarkerImage(cv::Mat &grey,int &nRotations);

};

/************************************
 *
 *
 *
 *
 ************************************/
/**
*/
template <typename CodeType> cv::Mat FiducidalMarkers<CodeType>::createMarkerImage(
        int id,
        int size,
        bool addWaterMark) throw (cv::Exception)
{
    cv::Mat marker(size,size, CV_8UC1);
    marker.setTo(cv::Scalar(0));

    int swidth=size/7;

    cv::Mat code(5,5, CV_8UC1);
    if(CodeType::encode(id, code) == -1){
        throw cv::Exception(9004,"id invalid","createMarker",__FILE__,__LINE__);
    }

    for (int y=0;y<5;y++) {
        for (int x=0;x<5;x++) {
            cv::Mat roi=marker(cv::Rect((x+1)* swidth,(y+1)* swidth,swidth,swidth));
            if ( code.at<uchar>(y, x) == 1 ) roi.setTo(cv::Scalar(255));
            else roi.setTo(cv::Scalar(0));
        }
    }

    if (addWaterMark){
        char idcad[30];
        sprintf(idcad,"#%d",id);
        float ax=float(size)/100.;
        cv::putText(
                    marker,
                    idcad,
                    cv::Point(0, marker.rows - marker.rows/40),
                    cv::FONT_HERSHEY_TRIPLEX,
                    ax*0.15f,
                    cv::Scalar::all(30)
        );
    }
    return marker;
}
/**
 *
 */
template <typename CodeType> cv::Mat FiducidalMarkers<CodeType>::getMarkerMat(int id) throw (cv::Exception)
{
    cv::Mat marker(5,5, CV_8UC1);
    if(CodeType::encode(id, marker) == -1) {
        throw cv::Exception (9189,"Invalid marker id","aruco::fiducidal::createMarkerMat",__FILE__,__LINE__);
    }
    return marker;
}
/************************************
 *
 *
 *
 *
 ************************************/

template <typename CodeType> cv::Mat FiducidalMarkers<CodeType>::createBoardImage(
        cv::Size gridSize,
        int MarkerSize,
        int MarkerDistance,
        BoardConfiguration& TInfo  ,
        std::vector<int> *excludedIds) throw (cv::Exception)
{



    srand(cv::getTickCount());
    int nMarkers=gridSize.height*gridSize.width;
    TInfo.resize(nMarkers);
    std::vector<int> ids=getListOfValidMarkersIds_random(nMarkers,excludedIds);
    for (int i=0;i<nMarkers;i++)
        TInfo[i].id=ids[i];

    int sizeY=gridSize.height*MarkerSize+(gridSize.height-1)*MarkerDistance;
    int sizeX=gridSize.width*MarkerSize+(gridSize.width-1)*MarkerDistance;
    //find the center so that the ref systeem is in it
    int centerX=sizeX/2;
    int centerY=sizeY/2;

    //indicate the data is expressed in pixels
    TInfo.mInfoType=BoardConfiguration::PIX;
    cv::Mat tableImage(sizeY,sizeX,CV_8UC1);
    tableImage.setTo(cv::Scalar(255));
    int idp=0;
    for (int y=0;y<gridSize.height;y++)
        for (int x=0;x<gridSize.width;x++,idp++) {
            cv::Mat subrect(tableImage,cv::Rect( x*(MarkerDistance+MarkerSize),y*(MarkerDistance+MarkerSize),MarkerSize,MarkerSize));
            cv::Mat marker=createMarkerImage( TInfo[idp].id,MarkerSize);
            //set the location of the corners
            TInfo[idp].resize(4);
            TInfo[idp][0]=cv::Point3f( x*(MarkerDistance+MarkerSize),y*(MarkerDistance+MarkerSize),0);
            TInfo[idp][1]=cv::Point3f( x*(MarkerDistance+MarkerSize)+MarkerSize,y*(MarkerDistance+MarkerSize),0);
            TInfo[idp][2]=cv::Point3f( x*(MarkerDistance+MarkerSize)+MarkerSize,y*(MarkerDistance+MarkerSize)+MarkerSize,0);
            TInfo[idp][3]=cv::Point3f( x*(MarkerDistance+MarkerSize),y*(MarkerDistance+MarkerSize)+MarkerSize,0);
            for (int i=0;i<4;i++) TInfo[idp][i]-=cv::Point3f(centerX,centerY,0);
            marker.copyTo(subrect);
        }

    return tableImage;
}

/************************************
 *
 *
 *
 *
 ************************************/
template <typename CodeType> cv::Mat  FiducidalMarkers<CodeType>::createBoardImage_ChessBoard(
        cv::Size gridSize,
        int MarkerSize,
        BoardConfiguration& TInfo,
        bool centerData,
        std::vector<int> *excludedIds) throw (cv::Exception)
{


    srand(cv::getTickCount());

    //determine the total number of markers required
    int nMarkers= 3*(gridSize.width*gridSize.height)/4;//overdetermine  the number of marker read
    std::vector<int> idsVector=getListOfValidMarkersIds_random(nMarkers,excludedIds);


    int sizeY=gridSize.height*MarkerSize;
    int sizeX=gridSize.width*MarkerSize;
    //find the center so that the ref systeem is in it
    int centerX=sizeX/2;
    int centerY=sizeY/2;

    cv::Mat tableImage(sizeY,sizeX,CV_8UC1);
    tableImage.setTo(cv::Scalar(255));
    TInfo.mInfoType=BoardConfiguration::PIX;
    int CurMarkerIdx=0;
    for (int y=0;y<gridSize.height;y++) {

        bool toWrite;
        if (y%2==0) toWrite=false;
        else toWrite=true;
        for (int x=0;x<gridSize.width;x++) {
            toWrite=!toWrite;
            if (toWrite) {
                if (CurMarkerIdx>=idsVector.size()) throw cv::Exception(999," FiducidalMarkers::createBoardImage_ChessBoard","INTERNAL ERROR. REWRITE THIS!!",__FILE__,__LINE__);
                TInfo.push_back( MarkerInfo(idsVector[CurMarkerIdx++]));

                cv::Mat subrect(tableImage, cv::Rect( x*MarkerSize,y*MarkerSize,MarkerSize,MarkerSize));
                cv::Mat marker=createMarkerImage( TInfo.back().id,MarkerSize);
                //set the location of the corners
                TInfo.back().resize(4);
                TInfo.back()[0]=cv::Point3f( x*(MarkerSize),y*(MarkerSize),0);
                TInfo.back()[1]=cv::Point3f( x*(MarkerSize)+MarkerSize,y*(MarkerSize),0);
                TInfo.back()[2]=cv::Point3f( x*(MarkerSize)+MarkerSize,y*(MarkerSize)+MarkerSize,0);
                TInfo.back()[3]=cv::Point3f( x*(MarkerSize),y*(MarkerSize)+MarkerSize,0);
                if (centerData) {
                    for (int i=0;i<4;i++)
                        TInfo.back()[i]-=cv::Point3f(centerX,centerY,0);
                }
                marker.copyTo(subrect);
            }
        }
    }

    return tableImage;
}



/************************************
 *
 *
 *
 *
 ************************************/
template <typename CodeType> cv::Mat  FiducidalMarkers<CodeType>::createBoardImage_Frame(
        cv::Size gridSize,
        int MarkerSize,
        int MarkerDistance,
        BoardConfiguration& TInfo ,
        bool centerData,
        std::vector<int> *excludedIds ) throw (cv::Exception)
{



    srand(cv::getTickCount());
    int nMarkers=2*gridSize.height*2*gridSize.width;
    std::vector<int> idsVector=getListOfValidMarkersIds_random(nMarkers,excludedIds);

    int sizeY=gridSize.height*MarkerSize+MarkerDistance*(gridSize.height-1);
    int sizeX=gridSize.width*MarkerSize+MarkerDistance*(gridSize.width-1);
    //find the center so that the ref systeem is in it
    int centerX=sizeX/2;
    int centerY=sizeY/2;

    cv::Mat tableImage(sizeY,sizeX,CV_8UC1);
    tableImage.setTo(cv::Scalar(255));
    TInfo.mInfoType=BoardConfiguration::PIX;
    int CurMarkerIdx=0;
    int mSize=MarkerSize+MarkerDistance;
    for (int y=0; y < gridSize.height; y++) {
        for (int x=0; x < gridSize.width; x++) {
            if (y==0 || y==gridSize.height-1 || x==0 ||  x==gridSize.width-1) {
                TInfo.push_back(  MarkerInfo(idsVector[CurMarkerIdx++]));
                cv::Mat subrect(tableImage,cv::Rect( x*mSize,y*mSize,MarkerSize,MarkerSize));
                cv::Mat marker=createMarkerImage( TInfo.back().id,MarkerSize);
                marker.copyTo(subrect);
                //set the location of the corners
                TInfo.back().resize(4);
                TInfo.back()[0]=cv::Point3f( x*(mSize),y*(mSize),0);
                TInfo.back()[1]=cv::Point3f( x*(mSize)+MarkerSize,y*(mSize),0);
                TInfo.back()[2]=cv::Point3f( x*(mSize)+MarkerSize,y*(mSize)+MarkerSize,0);
                TInfo.back()[3]=cv::Point3f( x*(mSize),y*(mSize)+MarkerSize,0);
                if (centerData) {
                    for (int i=0;i<4;i++)
                        TInfo.back()[i]-=cv::Point3f(centerX,centerY,0);
                }

            }
        }
    }

    return tableImage;
}


/************************************
 *
 *
 *
 *
 ************************************/
template <typename CodeType> int FiducidalMarkers<CodeType>::analyzeMarkerImage(
        cv::Mat &grey,
        int &nRotations)
{

    //Markers  are divided in 7x7 regions, of which the inner 5x5 belongs to marker info
    //the external border shoould be entirely black

    int swidth=grey.rows/7;
    for (int y=0;y<7;y++)
    {
        int inc=6;
        if (y==0 || y==6) inc=1;//for first and last row, check the whole border
        for (int x=0;x<7;x+=inc)
        {
            int Xstart=(x)*(swidth);
            int Ystart=(y)*(swidth);
            cv::Mat square=grey(cv::Rect(Xstart,Ystart,swidth,swidth));
            int nZ=countNonZero(square);
            if (nZ> (swidth*swidth) /2) {
                return -1;//can not be a marker because the border element is not black!
            }
        }
    }

    //now,
    std::vector<int> markerInfo(5);
    cv::Mat _bits=cv::Mat::zeros(5,5,CV_8UC1);
    cv::Mat bits = cv::Mat::zeros(5,5, CV_8UC1);
    //get information(for each inner square, determine if it is  black or white)

    for (int y=0;y<5;y++)
    {

        for (int x=0;x<5;x++)
        {
            int Xstart=(x+1)*(swidth);
            int Ystart=(y+1)*(swidth);
            cv::Mat square=grey(cv::Rect(Xstart,Ystart,swidth,swidth));
            int nZ=countNonZero(square);
            if (nZ> (swidth*swidth) /2)  _bits.at<uchar>( y,x)=1;
        }
    }

    nRotations = CodeType::rotate(_bits, bits);
    int id = CodeType::decode(bits);
    return id;
}



/************************************
 *
 *
 *
 *
 ************************************/
template <typename CodeType> int FiducidalMarkers<CodeType>::detect(const cv::Mat &in,int &nRotations)
{
    assert(in.rows==in.cols);
    cv::Mat grey;
    if ( in.type()==CV_8UC1) grey=in;
    else cv::cvtColor(in,grey,CV_BGR2GRAY);

    //threshold image
    cv::threshold(grey, grey,125, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);

    //now, analyze the interior in order to get the id
    //try first with the big ones

    return analyzeMarkerImage(grey,nRotations);;

}

template <typename CodeType> vector<int> FiducidalMarkers<CodeType>::getListOfValidMarkersIds_random(
        int nMarkers,
        std::vector<int> *excluded) throw (cv::Exception)
{

    if (excluded!=NULL)
        if (nMarkers+excluded->size()>1024) throw cv::Exception(8888,"FiducidalMarkers::getListOfValidMarkersIds_random","Number of possible markers is exceeded",__FILE__,__LINE__);

    std::vector<int> listOfMarkers(1024);
    //set a list with all ids
    for (int i=0;i<1024;i++) listOfMarkers[i]=i;

    if (excluded!=NULL)//set excluded to -1
        for (size_t i=0;i<excluded->size();i++)
            listOfMarkers[excluded->at(i)]=-1;
    //random shuffle
    random_shuffle(listOfMarkers.begin(),listOfMarkers.end());
    //now, take the first  nMarkers elements with value !=-1
    int i=0;
    std::vector<int> retList;
    while (retList.size()<nMarkers) {
        if (listOfMarkers[i]!=-1)
            retList.push_back(listOfMarkers[i]);
    i++;
    }
    return retList;
}


}

#endif
