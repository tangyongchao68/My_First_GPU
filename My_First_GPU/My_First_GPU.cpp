// My_First_GPU.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include <iostream>
//#include <opencv2\core\cuda.hpp>
//#include<opencv2\core.hpp>
//#include<highgui\highgui.hpp>
//#include<cv.hpp>
//#include<opencv.hpp>
#include<opencv2\opencv.hpp>


using namespace cv;
int main()
{

	int num_devices = cuda::getCudaEnabledDeviceCount();//获取显卡数量
	if (num_devices <= 0)
	{
		printf("没有显卡");
		return 0;
	}
	int enable_device_id = -1;//查找可以使用的显卡
	for (size_t i = 0; i < num_devices; i++)
	{
		cuda::DeviceInfo dev_info(i);
		if (dev_info.isCompatible())
		{
			enable_device_id = i;
		}
	}
	if (enable_device_id < 0)
	{
		printf("没有可使用显卡");
		return 0;
	}
	cuda::setDevice(enable_device_id);


//	Mat src_host = cv::imread("C:\\Users\\Administrator\\Desktop\\1.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src_host = cv::imread("./1.png", CV_LOAD_IMAGE_GRAYSCALE);
	long begintime = clock();
	cuda::GpuMat dst, src;
	src.upload(src_host);
	cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
	Mat result_host;
	dst.download(result_host);
	printf("花费时间：%ds", clock() - begintime);
	imshow("Result", result_host);

	waitKey();
	return 0;
}

