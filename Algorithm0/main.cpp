#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <chrono>

using namespace std;
using namespace chrono;

#define PI 3.1415926535897932384626433832
typedef std::chrono::high_resolution_clock Clock;

// 从文件中读取数据
vector<complex<double>> readDataFromFile(const string& filename)
{
    ifstream file(filename);
    vector<complex<double>> data;

    if (file) {
        double value;
        while (file >> value) {
            data.push_back(complex<double>(value, 0.0));
        }
        cout << "Data loaded from file: " << filename << endl;
    }
    else {
        cout << "Failed to open file: " << filename << endl;
    }

    file.close();
    return data;
}

// 将数据保存到文件
void saveDataToFile(const string& filename, const vector<complex<double>>& data)
{
    ofstream file(filename);
    if (file) {
        for (const complex<double>& sample : data) {
            file << sample.real() << endl;
        }
        cout << "Data saved to file: " << filename << endl;
    }
    else {
        cout << "Failed to create file: " << filename << endl;
    }

    file.close();
}

void fft0(vector<double> input, vector<complex<double>>& output)
{
    // 串行fft算法
    size_t length = input.size();
    if (length >= 2)
    {
        // 分为奇偶
        vector<double> odd;
        vector<double> even;
        for (size_t n = 0; n < length; n++)
        {
            if (n & 1)
            {
                odd.push_back(input.at(n));
            }
            else
            {
                even.push_back(input.at(n));
            }
        }
        // 重排
        // 低
        vector<complex<double>> fft_even_out(output.begin(), output.begin() + length / 2);
        // 高
        vector<complex<double>> fft_odd_out(output.begin() + length / 2, output.end());
        // 递归执行代码
        fft0(even, fft_even_out);
        fft0(odd, fft_odd_out);

        // 组合奇偶部分
        complex<double> odd_out;
        complex<double> even_out;
        for (size_t n = 0; n != length / 2; n++)
        {
            if (length == 2)
            {
                even_out = even[n] + fft_even_out[n];
                odd_out = odd[n] + fft_odd_out[n];
            }
            else
            {
                even_out = fft_even_out[n];
                odd_out = fft_odd_out[n];
            }
            // 翻转因子
            complex<double> w = exp(complex<double>(0, -2.0 * PI * double(n) / (double)(length)));
            // even part
            output[n] = even_out + w * odd_out;
            // odd part
            output[n + length / 2] = even_out - w * odd_out;
        }
    }
}

// FFT函数
void fft(vector<complex<double>>& input)
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
        complex<double> w_m(cos(2 * PI / k), -sin(2 * PI / k));

        for (int i = 0; i < n; i += k)
        {
            complex<double> w(1);
            for (int j = 0; j < m; j++)
            {
                complex<double> t = w * input[i + j + m];
                input[i + j + m] = input[i + j] - t;
                input[i + j] += t;
                w *= w_m;
            }
        }
    }
}

// 逆FFT函数
void ifft(vector<complex<double>>& input)
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
        complex<double> w_m(cos(2 * PI / k), sin(2 * PI / k));

        for (int i = 0; i < n; i += k)
        {
            complex<double> w(1);
            for (int j = 0; j < m; j++)
            {
                complex<double> t = w * input[i + j + m];
                input[i + j + m] = input[i + j] - t;
                input[i + j] += t;
                w *= w_m;
            }
        }
    }

    // 归一化
    for (complex<double>& sample : input)
    {
        sample /= n;
    }
}

vector<vector<complex<double>>> fft2D(const vector<vector<complex<double>>>& input)
{
    int rows = input.size();
    int cols = input[0].size();

    // 对行进行一维FFT
    vector<vector<complex<double>>> fftRows(rows);
    for (int i = 0; i < rows; i++)
    {
        fftRows[i] = input[i];
        fft(fftRows[i]);
    }

    // 对列进行一维FFT
    vector<vector<complex<double>>> fftCols(cols, vector<complex<double>>(rows));
    for (int j = 0; j < cols; j++)
    {
        vector<complex<double>> column(rows);
        for (int i = 0; i < rows; i++)
        {
            column[i] = fftRows[i][j];
        }
        fft(column);
        for (int i = 0; i < rows; i++)
        {
            fftCols[j][i] = column[i];
        }
    }

    return fftCols;
}

// 信号快速相关计算
vector<complex<double>> fastCorrelation(const vector<complex<double>>& signal1, const vector<complex<double>>& signal2)
{
    // 信号长度
    int n = signal1.size();
    // 补零使两个信号长度相同，并保证长度为2的幂次
    int fftSize = 1;
    while (fftSize < n * 2) {
        fftSize <<= 1;
    }
    // 对信号进行FFT变换
    vector<complex<double>> fftSignal1(fftSize, 0.0);
    vector<complex<double>> fftSignal2(fftSize, 0.0);

    for (int i = 0; i < n; i++) {
        fftSignal1[i] = signal1[i];
        fftSignal2[i] = signal2[i];
    }
    fft(fftSignal1);
    fft(fftSignal2);
    // 计算频域中两个信号的乘积
    vector<complex<double>> fftResult(fftSize, 0.0);
    for (int i = 0; i < fftSize; i++) {
        fftResult[i] = fftSignal1[i] * conj(fftSignal2[i]);
    }
    // 对乘积结果进行逆FFT变换
    ifft(fftResult);
    // 返回逆FFT结果
    return fftResult;
}


// 普通卷积函数
vector<double> convolution(const vector<double>& signal1, const vector<double>& signal2)
{
    int n1 = signal1.size();
    int n2 = signal2.size();
    int resultSize = n1 + n2 - 1;
    vector<double> result(resultSize, 0.0);

    for (int i = 0; i < resultSize; i++) {
        for (int j = 0; j <= i; j++) {
            if (j < n1 && (i - j) < n2) {
                result[i] += signal1[j] * signal2[i - j];
            }
        }
    }
    return result;
}


// 快速卷积
vector<complex<double>> fastConvolution(const vector<complex<double>>& signal1, const vector<complex<double>>& signal2)
{
    // 信号长度
    int n = signal1.size();

    // 补零使两个信号长度相同，并保证长度为2的幂次
    int fftSize = 1;
    while (fftSize < n * 2) {
        fftSize <<= 1;
    }

    // 对信号进行FFT变换
    vector<complex<double>> fftSignal1(fftSize, 0.0);
    vector<complex<double>> fftSignal2(fftSize, 0.0);

    for (int i = 0; i < n; i++) {
        fftSignal1[i] = signal1[i];
        fftSignal2[i] = signal2[i];
    }

    fft(fftSignal1);
    fft(fftSignal2);

    // 计算频域中两个信号的乘积
    vector<complex<double>> fftResult(fftSize, 0.0);
    for (int i = 0; i < fftSize; i++) {
        fftResult[i] = fftSignal1[i] * fftSignal2[i];
    }

    // 对乘积结果进行逆FFT变换
    ifft(fftResult);

    // 返回逆FFT结果
    return fftResult;
}


// 大数乘法
vector<int> multiply(const vector<int>& num1, const vector<int>& num2)
{
    int n = num1.size();
    int m = num2.size();
    int fftSize = 1;
    while (fftSize < n + m) fftSize <<= 1;

    vector<complex<double>> fftNum1(fftSize, 0);
    vector<complex<double>> fftNum2(fftSize, 0);

    // 将输入的大数转换为复数形式
    for (int i = 0; i < n; i++)
    {
        fftNum1[i] = num1[i];
    }
    for (int i = 0; i < m; i++)
    {
        fftNum2[i] = num2[i];
    }

    // 进行 FFT 变换
    fft(fftNum1);
    fft(fftNum2);

    // 点乘
    vector<complex<double>> fftResult(fftSize);
    for (int i = 0; i < fftSize; i++)
    {
        fftResult[i] = fftNum1[i] * fftNum2[i];
    }

    // 进行逆 FFT 变换
    ifft(fftResult);

    // 提取实部的整数部分作为乘积结果
    vector<int> result(n + m - 1);
    for (int i = 0; i < n + m - 1; i++)
    {
        result[i] = round(fftResult[i].real());
    }

    return result;
}


//int main()
//{
//    vector<int> num1 = { 6,6,6,6 }; // 第一个大数
//    vector<int> num2 = { 4, 5, 6 }; // 第二个大数
//
//    vector<int> product = multiply(num1, num2);
//
//    // 打印乘积结果
//    for (int i = 0; i < product.size(); i++)
//    {
//        cout << product[i] << " ";
//    }
//    cout << endl;
//
//    return 0;
//}



int main()
{
    // 输入两个信号
    vector<complex<double>> signal1 = readDataFromFile("fft_8192.txt");
    vector<complex<double>> signal2 = readDataFromFile("fft_8192.txt");

    ifstream fi("fft_8192.txt");
    vector<double> data;
    string read_temp;
    while (fi.good())
    {
        getline(fi, read_temp);
        data.push_back(stod(read_temp));
    }

    // 进行信号的快速相关计算
    // vector<complex<double>> correlationResult = fastCorrelation(signal1, signal2);

    // 保存相关计算结果到文件
    // saveDataToFile("correlation_result.txt", correlationResult);

    // 打印结果
    //cout << "Correlation Result: correlation_result.txt" << endl;
    //for (const complex<double>& sample : correlationResult) {
    //    cout << sample << endl;
    //}

    // 进行信号的快速卷积
    auto t1 = Clock::now();
    vector<double> normalconvolutionResult = convolution(data, data);
    auto t2 = Clock::now();// 计时结束
    cout << "convolution cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";


    t1 = Clock::now();
    vector<complex<double>> convolutionResult = fastConvolution(signal1, signal2);
    t2 = Clock::now();// 计时结束
    cout << "fastConvolution cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";
    // 保存卷积结果到文件
    saveDataToFile("convolution_result.txt", convolutionResult);

    //vector<complex<double>> num1 = { 1, 2, 3 }; // 第一个大数
    //vector<complex<double>> num2 = { 4, 5, 6 }; // 第二个大数

    //vector<complex<double>> result = multiply(num1, num2);

    //// 输出结果
    //cout << "Multiplication Result: ";
    //for (const complex<double>& num : result)
    //{
    //    cout << num.real() << " ";
    //}
    //cout << endl;

    return 0;
}
//


//int main()
//{
//    // 二维数据
//    vector<vector<complex<double>>> input = {
//        {1, 2, 3, 4},
//        {4, 5, 6, 7},
//        {7, 8, 9, 10},
//        {11,12,13,14}
//    };
//
//    // 进行二维FFT
//    vector<vector<complex<double>>> result = fft2D(input);
//
//    // 打印结果
//    for (int i = 0; i < result.size(); i++)
//    {
//        for (int j = 0; j < result[i].size(); j++)
//        {
//            cout << result[i][j] << " ";
//        }
//        cout << endl;
//    }
//
//    return 0;
//}