#include <iostream>  
#include <string>  
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
int adjustBrightnessContrast(Mat& src, Mat& dst, int brightness, int contrast)  
{  
    //Mat input = src.getMat();  
    //if( input.empty() ) {  
    //    return -1;  
    //}  
  
    dst.create(src.size(), src.type());  
    //Mat output = dst.getMat();  
  
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
  
    LUT(src, lookupTable, dst);  
  
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

void ColorBalance(Mat& img, Mat& cbImg, int cR, int cG, int cB)  
{  
    if ( cbImg.empty())   
        cbImg.create(img.rows, img.cols, img.type());    
  
    //cbImg = cv::Scalar::all(0);  
  
    int i, j;  
    Size size = img.size();  
    int chns = img.channels();  
  
    if (img.isContinuous() && cbImg.isContinuous())  
    {   
        size.width *= size.height;   
        size.height = 1;  
    }   
    
    // 验证参数范围  
    if ( cR<-255 )   
        cR = -255;  
  
    if ( cG<-255 )   
        cG = -255;  
  
    if ( cB<-255 )   
        cB = -255;  
  
    if ( cR>255)  
        cR = 255;  
  
    if ( cG>255)  
        cG = 255;  
  
    if ( cB>255)  
        cB = 255;  
  
  
    for (  i= 0; i<size.height; ++i)  
    {   
        const unsigned char* src = (const unsigned char*)(img.data+ img.step*i);  
        unsigned char* dst = (unsigned char*)cbImg.data+cbImg.step*i;  
        for (  j=0; j<size.width; ++j)  
        {
            dst[j*chns] = saturate_cast<uchar>(src[j*chns] +cR);  
            dst[j*chns+1] = saturate_cast<uchar>(src[j*chns+1] +cG);  
            dst[j*chns+2] = saturate_cast<uchar>(src[j*chns+2] +cB);    
        }
    }    
}  

// Gamma 矫正量[0.1, 5.0]  
void GammaCorrect(Mat& img, Mat& cImg, float ga)  
{  
    if ( cImg.empty())    
        cImg.create(img.rows, img.cols, img.type());          
  
    //cImg = cv::Scalar::all(0);  
  
    int i, j;  
    Size size = img.size();  
    int chns = img.channels();  
  
    if (img.isContinuous() && cImg.isContinuous())  
    {  
        size.width *= size.height;   
        size.height = 1;  
    }  
  
    // 验证参数范围  
    ga = ga / 10.0;
    if ( ga<0.1) ga = -0.1;  
    if ( ga> 5.0) ga = 5.0;      
  
    // 加速，建立查找表  
    unsigned char lut[256];    
    for(  i = 0; i < 256; i++ )    
    {    
        lut[i] = saturate_cast<uchar>(cv::pow((float)(i/255.0), ga) * 255.0f);    
    }   
  
    for (  i= 0; i<size.height; ++i)  
    {  
        const unsigned char* src = (const unsigned char*)(img.data+ img.step*i);  
        unsigned char* dst = (unsigned char*)cImg.data+cImg.step*i;  
        for (  j=0; j<size.width; ++j)  
        {                     
            dst[j*chns] = lut[src[j*chns]];  
            dst[j*chns+1] = lut[src[j*chns+1]];  
            dst[j*chns+2] = lut[src[j*chns+2]];       
        }  
    }     
} 


  

//=====主程序开始====  
  
static string window_name = "control";  
static string window_img = "image";  
static Mat src;  
static Mat dst;  
static Mat mask;  
static int brightness = 255;  
static int contrast = 255;  
  
static int hue = 180;  
static int saturation = 255;  
static int ilumination = 255;  

static int cR = 255;
static int cG = 255;
static int cB = 255;

static int ga= 10;


static void callbackAdjust_bright(int , void *)  
{  
    adjustBrightnessContrast(src, dst, brightness - 255, contrast - 255);  
    imshow(window_img, dst);  
}  
  
static void callbackAdjust_HSI(int , void *)  
{  
    AdjustHSI(src, dst, hue-180, saturation-255, ilumination-255);
    imshow(window_img, dst);  
}  
  
static void callbackAdjust_ColorBalance(int , void *)  
{  
    ColorBalance(src, dst, cB - 255, cG - 255, cR - 255);
    imshow(window_img, dst);  
}  


int getMask(Mat& img, Mat& mask)
{
    Size size = img.size();  
    int chns = img.channels();  

    for (int i= 0; i<size.height; ++i)  
    {   
        const unsigned char* src = (const unsigned char*)(img.data+ img.step*i);  
        unsigned char* dst = (unsigned char*)mask.data+mask.step*i;  
        for (int j=0; j<size.width; ++j)  
        {
            if (saturate_cast<uchar>(src[j*chns])<200 || saturate_cast<uchar>(src[j*chns+1])<200 || saturate_cast<uchar>(src[j*chns+2])<200)
            {
                dst[j*chns] = 255;  
                dst[j*chns+1] = 255;  
                dst[j*chns+2] = 255;       
            }
        }
    }    

    return 0;
}


int* getRGB(Mat& img)
{
    int *rgb = new int[3];
    for (int i=0; i<3; i++)
    {
        rgb[i] = 0;
    }

    Size size = img.size();  
    int chns = img.channels();  

    mask.create(img.size(), img.type());  
    getMask(src, mask);
    Mat grayMask = Mat::zeros(img.size(), CV_8UC1);
    cvtColor(mask, grayMask, CV_BGR2GRAY);      

    CvScalar cs;  
    cs = mean(img, grayMask);  

    rgb[0] = cs.val[2];
    rgb[1] = cs.val[1];
    rgb[2] = cs.val[0];

    return rgb;
}
  
int* getLAB(Mat& img)
{
    int *lab = new int[3];
    for (int i=0; i<3; i++)
    {
        lab[i] = 0;
    }

    Mat temp;  
    temp.create(img.rows, img.cols, img.type());      
    cvtColor(img, temp, CV_BGR2Lab);      

    Size size = img.size();  
    int chns = img.channels();  

    mask = Mat::zeros(img.size(), img.type());
    getMask(src, mask);
    Mat grayMask = Mat::zeros(img.size(), CV_8UC1);
    cvtColor(mask, grayMask, CV_BGR2GRAY);      
    //imwrite("grayMask.jpg", grayMask);

    CvScalar cs;  
    cs = mean(temp, grayMask);  

    lab[0] = cs.val[0];
    lab[1] = cs.val[1];
    lab[2] = cs.val[2];

    if ( temp.empty())  
        temp.release();  
      
    return lab;
}
  
int* getHSV(Mat& img)
{
    int *hsv = new int[3];
    for (int i=0; i<3; i++)
    {
        hsv[i] = 0;
    }

    Mat temp;  
    temp.create(img.rows, img.cols, img.type());      
    cvtColor(img, temp, CV_BGR2HSV);      

    Size size = img.size();  
    int chns = img.channels();  

    mask = Mat::zeros(img.size(), img.type());
    getMask(src, mask);
    Mat grayMask = Mat::zeros(img.size(), CV_8UC1);
    cvtColor(mask, grayMask, CV_BGR2GRAY);      
    //imwrite("grayMask.jpg", grayMask);

    CvScalar cs;  
    cs = mean(temp, grayMask);  

    hsv[0] = cs.val[0];
    hsv[1] = cs.val[1];
    hsv[2] = cs.val[2];

    if ( temp.empty())  
        temp.release();  
      
    return hsv;
}
  
static void callbackAdjust(int , void *)  
{  
    adjustBrightnessContrast(src, dst, brightness - 255, contrast - 255);  
    AdjustHSI(dst, dst, hue-180, saturation-255, ilumination-255);
    ColorBalance(dst, dst, cB - 255, cG - 255, cR - 255);
    GammaCorrect(dst, dst, ga);
    
    int *rgb = NULL;
    rgb = getRGB(dst);
    Point ptRGB(5,dst.size().height/10);
    stringstream ss;
    ss << "RGB:" << rgb[0] << "," << rgb[1] << "," << rgb[2];
    string strRGB = ss.str();
    putText(dst,strRGB,ptRGB,CV_FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1,1);
    cout << strRGB << endl;
    delete rgb;
    rgb = NULL;

    int *lab = NULL;
    lab = getLAB(dst);
    Point ptLAB(5,dst.size().height*3/10);
    stringstream ssLAB;
    ssLAB << "LAB:" << lab[0] << "," << lab[1] << "," << lab[2];
    string strLAB = ssLAB.str();
    //putText(dst,strLAB,ptRGB,CV_FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1,1);
    cout << strLAB << endl;
    delete lab;
    lab = NULL;

    int *hsv = NULL;
    hsv = getHSV(dst);
    Point ptHSV(5,dst.size().height*6/10);
    stringstream ssHSV;
    ssHSV << "HSV:" << hsv[0] << "," << hsv[1] << "," << hsv[2];
    string strHSV = ssHSV.str();
    //putText(dst,strHSV,ptRGB,CV_FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1,1);
    cout << strHSV << endl << endl;
    delete hsv;
    hsv = NULL;

    imshow(window_img, dst);  
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
    dst.create(src.size(), src.type());  
  
    namedWindow(window_name, CV_WINDOW_NORMAL| CV_WINDOW_KEEPRATIO| CV_GUI_EXPANDED);  
    namedWindow(window_img, CV_WINDOW_NORMAL| CV_WINDOW_KEEPRATIO| CV_GUI_EXPANDED);  
    resizeWindow(window_img, 1024, 1080);
    createTrackbar("brightness", window_name, &brightness, 2*brightness, callbackAdjust);  
    createTrackbar("contrast", window_name, &contrast, 2*contrast, callbackAdjust);  

    createTrackbar("hue", window_name, &hue, 2*hue, callbackAdjust);  
    createTrackbar("saturation", window_name, &saturation, 2*saturation, callbackAdjust);  
    createTrackbar("ilumination", window_name, &ilumination, 2*ilumination, callbackAdjust);  

    createTrackbar("cR", window_name, &cR, 2*cR, callbackAdjust);
    createTrackbar("cG", window_name, &cG, 2*cG, callbackAdjust);
    createTrackbar("cB", window_name, &cB, 2*cB, callbackAdjust);

    createTrackbar("gamma", window_name, &ga, 50, callbackAdjust);

    callbackAdjust(0, 0);  
  
    waitKey();  

    return 0;  
  
}  
