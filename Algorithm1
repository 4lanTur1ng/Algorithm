#include <iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;
const int height = 512, width = 512;

const double PI = acos(-1); // pi值


struct Cpx // 定义一个复数结构体和复数运算法则
{
	double r, i;
	Cpx() : r(0), i(0) {}
	Cpx(double _r, double _i) : r(_r), i(_i) {}
};
Cpx operator + (Cpx a, Cpx b) { return Cpx(a.r + b.r, a.i + b.i); }
Cpx operator - (Cpx a, Cpx b) { return Cpx(a.r - b.r, a.i - b.i); }
Cpx operator * (Cpx a, Cpx b) { return Cpx(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); }
Cpx operator / (Cpx a, int b) { return Cpx(a.r * 1.0 / b, a.i * 1.0 / b); }

Mat BGR2GRAY(Mat img)
{
	int w = img.cols;
	int h = img.rows;
	Mat grayImg(h, w, CV_8UC1);

	uchar* p = grayImg.ptr<uchar>(0);
	Vec3b* pImg = img.ptr<Vec3b>(0);

	for (int i = 0; i < w * h; ++i)
	{
		p[i] = 0.2126 * pImg[i][2] + 0.7152 * pImg[i][1] + 0.0722 * pImg[i][0];
	}
	return grayImg;
}

Mat Resize(Mat img)
{
	int w = img.cols;
	int h = img.rows;
	Mat out(height, width, CV_8UC1);
	uchar* p = out.ptr<uchar>(0);
	uchar* p2 = img.ptr<uchar>(0);
	int x_before, y_before;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{
			x_before = (int)x * w * 1.0 / width;
			y_before = (int)y * h * 1.0 / height;
			p[y * width + x] = p2[y_before * w + x_before];
		}
	}
	return out;

}

//自动缩放到0-255范围内,并变换象限，将低频移至中间
int AutoScale(Mat src, Mat out)
{
	int w = src.cols;
	int h = src.rows;
	double* p = src.ptr<double>(0);
	uchar* pOut = out.ptr<uchar>(0);
	double max = p[0];
	double min = p[0];
	for (int i = 0; i < w * h; i++)
	{
		if (p[i] > max) max = p[i];
		if (p[i] < min) min = p[i];
	}

	double scale = 255.0 / (max - min);

	for (int i = 0; i < w * h; i++)
	{
		int j = i + w * h / 2 + w / 2;
		if (j > w * h) j = j - w * h;   //低频移至中间
		pOut[i] = (uchar)((p[j] - min) * scale);
	}
	return 0;
}

void fft(vector<Cpx>& a, int lim, int opt)
{
	if (lim == 1) return;
	vector<Cpx> a0(lim >> 1), a1(lim >> 1); // 初始化一半大小，存放偶数和奇数部分
	for (int i = 0; i < lim; i += 2)
		a0[i >> 1] = a[i], a1[i >> 1] = a[i + 1]; // 分成偶数部分和奇数部分

	fft(a0, lim >> 1, opt); // 递归计算偶数部分
	fft(a1, lim >> 1, opt); // 递归计算偶数部分

	Cpx wn(cos(2 * PI / lim), opt * -sin(2 * PI / lim)); //等于WN
	Cpx w(1, 0);
	for (int k = 0; k < (lim >> 1); k++) // 见蝶形图1运算过程
	{
		a[k] = a0[k] + w * a1[k];
		a[k + (lim >> 1)] = a0[k] - w * a1[k];
		w = w * wn;
	}

	//for (int k = 0; k < (lim >> 1); k++) // 见蝶形图2，小优化一下，少一次乘法
	//{
	//	Cpx t = w * a1[k];
	//	a[k] = a0[k] + t;
	//	a[k + (lim >> 1)] = a0[k] - t;
	//	w = w * wn;
	//}

}

//二进制逆序排列
int ReverseBin(int a, int n)
{
	int ret = 0;
	for (int i = 0; i < n; i++)
	{
		if (a & (1 << i)) ret |= (1 << (n - 1 - i));
	}
	return ret;
}

void fft2(vector<Cpx>& a, int lim, int opt)
{
	int index;
	vector<Cpx> tempA(lim);
	for (int i = 0; i < lim; i++)
	{
		index = ReverseBin(i, log2(lim));
		tempA[i] = a[index];
	}

	vector<Cpx> WN(lim / 2);
	//生成WN表,避免重复计算
	for (int i = 0; i < lim / 2; i++)
	{
		WN[i] = Cpx(cos(2 * PI * i / lim), opt * -sin(2 * PI * i / lim));
	}

	//蝶形运算
	int Index0, Index1;
	Cpx temp;
	for (int steplenght = 2; steplenght <= lim; steplenght *= 2)
	{
		for (int step = 0; step < lim / steplenght; step++)
		{
			for (int i = 0; i < steplenght / 2; i++)
			{
				Index0 = steplenght * step + i;
				Index1 = steplenght * step + i + steplenght / 2;

				temp = tempA[Index1] * WN[lim / steplenght * i];
				tempA[Index1] = tempA[Index0] - temp;
				tempA[Index0] = tempA[Index0] + temp;
			}
		}
	}
	for (int i = 0; i < lim; i++)
	{
		if (opt == -1)
		{
			a[i] = tempA[i] / lim;
		}
		else
		{
			a[i] = tempA[i];
		}
	}
}


void fft_recur(vector<Cpx>& input, int lim, int opt)//迭代算法 
{
	int n = input.size();

	// 数据重排
	for (int i = 1, j = 0; i < n; i++)
	{
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// 蝴蝶运算
	for (int k = 2; k <= n; k <<= 1)
	{
		int m = k >> 1;
		Cpx w_m(cos(PI / m), -1*opt*sin(PI / m));

		for (int i = 0; i < n; i += k)
		{
			Cpx w(1,0);
			for (int j = 0; j < m; j++)
			{
				Cpx t = w * input[i + j + m];
				input[i + j + m] = input[i + j] - t;
				input[i + j] = input[i + j]+ t;
				w = w *w_m;
			}
		}
	}

	
	for (int i = 0; i < lim; i++)
	{
		if (opt == -1)
		{
			input[i] = input[i] / lim;
		}
		else
		{
			input[i] = input[i];
		}
	}
	
}
void FFT2D(Cpx(*src)[width], Cpx(*dst)[width], int opt)
{
	//第一遍fft
	for (int i = 0; i < height; i++)
	{
		vector<Cpx> tempData(width);
		//获取每行数据
		for (int j = 0; j < width; j++)
		{
			tempData[j] = src[i][j];
		}
		//一维FFT
		fft_recur(tempData, width, opt);
		//写入每行数据
		for (int j = 0; j < width; j++)
		{
			dst[i][j] = tempData[j];
		}
	}

	//第二遍fft
	for (int i = 0; i < width; i++)
	{
		vector<Cpx> tempData(height);
		//获取每列数据
		for (int j = 0; j < height; j++)
		{
			tempData[j] = dst[j][i];
		}
		//一维FFT
		fft_recur(tempData, height, opt);
		//写入每列数据
		for (int j = 0; j < height; j++)
		{
			dst[j][i] = tempData[j];
		}
	}
}

void Mat2Cpx(Mat src, Cpx(*dst)[width])
{
	//这里Mat里的数据得是unchar类型
	uchar* p = src.ptr<uchar>(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst[i][j] = Cpx(p[i * width + j], 0);
		}
	}
}

void Cpx2Mat(Cpx(*src)[width], Mat dst)
{
	//这里Mat里的数据得是unchar类型
	uchar* p = dst.ptr<uchar>(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double g = sqrt(src[i][j].r * src[i][j].r);
			p[j + i * width] = (uchar)g;
		}
	}
}

void Cpx2MatDouble(Cpx(*src)[width], Mat dst)
{
	//这里Mat里的数据得是double类型
	double* p = dst.ptr<double>(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double g = sqrt(src[i][j].r * src[i][j].r + src[i][j].i * src[i][j].i);
			g = log(g + 1);  //转换为对数尺度
			p[j + i * width] = (double)g;
		}
	}
}

Mat sharpenImage(const cv::Mat& image)
{
	// 创建拉普拉斯滤波器
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		0, 1, 0,
		1, -4, 1,
		0, 1, 0);

	// 对图像进行卷积操作
	cv::Mat sharpenedImage;
	cv::filter2D(image, sharpenedImage, CV_8UC3, kernel);

	return sharpenedImage;
}


Mat blurImage(const cv::Mat& image)
{
	// 应用高斯滤波器
	cv::Mat blurredImage;
	cv::GaussianBlur(image, blurredImage, cv::Size(5,5), 0);

	return blurredImage;
}
int main()
{
	Mat img = imread("..\\curry.png");
	imshow("img1", img);

	Mat gray = BGR2GRAY(img);
	//imshow("gray", gray);
	imwrite("gray1.jpg", gray);

	Mat imgRez = Resize(gray);
	//imshow("imgRez", imgRez);
	imwrite("imgRez1.jpg", imgRez);

	Mat sharpenedImage = sharpenImage(img);
	Mat blurredImage = blurImage(img);
	imwrite("sharp1.jpg", sharpenedImage);
	imwrite("blur1.jpg", blurredImage);
	
	imshow("Blurred Image", blurredImage);

	imshow("Sharpened Image", sharpenedImage);

	Cpx(*src)[width] = new Cpx[height][width];

	Mat2Cpx(imgRez, src);//把imgRez值转到src

	Cpx(*dst)[width] = new Cpx[height][width];



	double t1 = getTickCount();
	FFT2D(src, dst, 1);//正变换结果在dst
	double t2 = getTickCount();

	double t1t2 = (t2 - t1) / getTickFrequency();
	std::cout << "DFT耗时: " << t1t2 << "秒" << std::endl;


	Cpx(*dst2)[width] = new Cpx[height][width];
	double t3 = getTickCount();
	FFT2D(dst, dst2, -1);//逆变换结果保存在dst2
	double t4 = getTickCount();

	double t3t4 = (t4 - t3) / getTickFrequency();
	std::cout << "DFT耗时: " << t3t4 << "秒" << std::endl;

	Mat out2 = Mat::zeros(height, width, CV_8UC1);
	Cpx2Mat(dst2, out2);
	imshow("out2", out2);
	//imwrite("out2.jpg", out2);

	Mat out = Mat::zeros(height, width, CV_64F);
	Cpx2MatDouble(dst, out);
	Mat out3 = Mat::zeros(height, width, CV_8UC1);

	AutoScale(out, out3);
	imshow("out3", out3);
	//imwrite("out3.jpg", out3);


	waitKey(0);
}


