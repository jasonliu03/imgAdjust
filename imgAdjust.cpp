#include <iostream>  
#include "opencv2/core.hpp"  
#include "opencv2/imgproc.hpp"  
#include "opencv2/highgui.hpp"  
  
using namespace std;  
using namespace cv;  
  
  
#define SWAP(a, b, t)  do { t = a; a = b; b = t; } while(0)  
#define CLIP_RANGE(value, min, max)  ( (value) > (max) ? (max) : (((value) < (min)) ? (min) : (value)) )  
#define COLOR_RANGE(value)  CLIP_RANGE(value, 0, 255)  
  
/** 
 * Adjust Brightness and Contrast 
 * 
 * @param src [in] InputArray 
 * @param dst [out] OutputArray 
 * @param brightness [in] integer, value range [-255, 255] 
 * @param contrast [in] integer, value range [-255, 255] 
 * 
 * @return 0 if success, else return error code 
 */  
int adjustBrightnessContrast(InputArray src, OutputArray dst, int brightness, int contrast)  
{  
    Mat input = src.getMat();  
    if( input.empty() ) {  
        return -1;  
    }  
  
    dst.create(src.size(), src.type());  
    Mat output = dst.getMat();  
  
    brightness = CLIP_RANGE(brightness, -255, 255);  
    contrast = CLIP_RANGE(contrast, -255, 255);  
  
    /** 
    Algorithm of Brightness Contrast transformation 
    The formula is: 
        y = [x - 127.5 * (1 - B)] * k + 127.5 * (1 + B); 
 
        x is the input pixel value 
        y is the output pixel value 
        B is brightness, value range is [-1,1] 
        k is used to adjust contrast 
            k = tan( (45 + 44 * c) / 180 * PI ); 
            c is contrast, value range is [-1,1] 
    */  
  
    double B = brightness / 255.;  
    double c = contrast / 255. ;  
    double k = tan( (45 + 44 * c) / 180 * M_PI );  
  
    Mat lookupTable(1, 256, CV_8U);  
    uchar *p = lookupTable.data;  
    for (int i = 0; i < 256; i++)  
        p[i] = COLOR_RANGE( (i - 127.5 * (1 - B)) * k + 127.5 * (1 + B) );  
  
    LUT(input, lookupTable, output);  
  
    return 0;  
}  


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
              
            val = src[j*chns+2]+ilumination;  
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
static int brightness = 255;  
static int contrast = 255;  
  
static int hue = 180;  
static int saturation = 255;  
static int ilumination = 255;  

static void callbackAdjust_bright(int , void *)  
{  
    Mat dst;  
    adjustBrightnessContrast(src, dst, brightness - 255, contrast - 255);  
    imshow(window_name, dst);  
}  
  
static void callbackAdjust_HSI(int , void *)  
{  
    Mat dst;  
    AdjustHSI(src, dst, hue-180, saturation-255, ilumination-255);
    imshow(window_name, dst);  
}  
  
  
int main(int argc, char** argv)  
{  
    char * filename = "test.jpg";
    if(argc > 1)
    {   
        filename = argv[1];
    } 

    src = imread(filename);  
  
    if ( !src.data ) {  
        cout << "error read image" << endl;  
        return -1;  
    }  
  
    namedWindow(window_name, CV_WINDOW_NORMAL| CV_WINDOW_KEEPRATIO| CV_GUI_EXPANDED);  
    createTrackbar("brightness", window_name, &brightness, 2*brightness, callbackAdjust_bright);  
    createTrackbar("contrast", window_name, &contrast, 2*contrast, callbackAdjust_bright);  

    createTrackbar("hue", window_name, &hue, 2*hue, callbackAdjust_HSI);  
    createTrackbar("saturation", window_name, &saturation, 2*saturation, callbackAdjust_HSI);  
    createTrackbar("ilumination", window_name, &ilumination, 2*ilumination, callbackAdjust_HSI);  
    callbackAdjust_bright(0, 0);  
    callbackAdjust_HSI(0, 0);  
  
    waitKey();  

    return 0;  
  
}  
