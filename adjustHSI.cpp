#include <iostream>  
#include "opencv2/core.hpp"  
#include "opencv2/imgproc.hpp"  
#include "opencv2/highgui.hpp"  
  
using namespace std;  
using namespace cv;  

// H:0~180, S:0~255, V:0~255  
void AdjustHSI(Mat& img, Mat& aImg, int  hue, int saturation, int ilumination)  
{  
    if ( aImg.empty())    
        aImg.create(img.rows, img.cols, img.type());      
  
    Mat temp;  
    temp.create(img.rows, img.cols, img.type());      
  
    cvtColor(img, temp, CV_RGB2HSV);      
  
    int i, j;  
    Size size = img.size();  
    int chns = img.channels();  
  
    if (temp.isContinuous())  
    {  
        size.width *= size.height;   
        size.height = 1;  
    }  
  
    // 验证参数范围  
    if ( hue<-180 )  
        hue = -180;  
  
    if ( saturation<-255)  
        saturation = -255;  
  
    if ( ilumination<-255 )  
        ilumination = -255;  
  
    if ( hue>180)  
        hue = 180;  
  
    if ( saturation>255)  
        saturation = 255;  
  
    if ( ilumination>255)  
        ilumination = 255;  
  
  
    for (  i= 0; i<size.height; ++i)  
    {         
        unsigned char* src = (unsigned char*)temp.data+temp.step*i;  
        for (  j=0; j<size.width; ++j)  
        {  
            float val = src[j*chns]+hue;  
            if ( val < 0) val = 0.0;  
            if ( val > 180 ) val = 180;  
            src[j*chns] = val;  
  
            val = src[j*chns+1]+saturation;  
            if ( val < 0) val = 0;  
            if ( val > 255 ) val = 255;  
            src[j*chns+1] = val;  
              
            val = src[j*chns+2]+saturation;  
            if ( val < 0) val = 0;  
            if ( val > 255 ) val = 255;  
            src[j*chns+2] = val;                          
        }  
    }     
  
    cvtColor(temp, aImg, CV_HSV2RGB);  
    if ( temp.empty())  
        temp.release();  
      
}  

  
//=====主程序开始====  
  
static string window_name = "photo";  
static Mat src;  
static int hue = 180;  
static int saturation = 255;  
static int ilumination = 255;  
  
static void callbackAdjust(int , void *)  
{  
    Mat dst;  
    //adjustBrightnessContrast(src, dst, brightness - 255, contrast - 255);  
    AdjustHSI(src, dst, hue-180, saturation-255, ilumination-255);
    imshow(window_name, dst);  
}  
  
  
int main()  
{  
    src = imread("05.jpg");  
  
    if ( !src.data ) {   
        cout << "error read image" << endl;  
        return -1;  
    }   
  
    namedWindow(window_name, CV_WINDOW_NORMAL| CV_WINDOW_KEEPRATIO| CV_GUI_EXPANDED);  
    createTrackbar("hue", window_name, &hue, 2*hue, callbackAdjust);  
    createTrackbar("saturation", window_name, &saturation, 2*saturation, callbackAdjust);  
    createTrackbar("ilumination", window_name, &ilumination, 2*ilumination, callbackAdjust);  
    callbackAdjust(0, 0);  
  
    waitKey();  
  
    return 0;  
  
} 
