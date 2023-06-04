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

// ���ļ��ж�ȡ����
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

// �����ݱ��浽�ļ�
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
    // ����fft�㷨
    size_t length = input.size();
    if (length >= 2)
    {
        // ��Ϊ��ż
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
        // ����
        // ��
        vector<complex<double>> fft_even_out(output.begin(), output.begin() + length / 2);
        // ��
        vector<complex<double>> fft_odd_out(output.begin() + length / 2, output.end());
        // �ݹ�ִ�д���
        fft0(even, fft_even_out);
        fft0(odd, fft_odd_out);

        // �����ż����
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
            // ��ת����
            complex<double> w = exp(complex<double>(0, -2.0 * PI * double(n) / (double)(length)));
            // even part
            output[n] = even_out + w * odd_out;
            // odd part
            output[n + length / 2] = even_out - w * odd_out;
        }
    }
}

// FFT����
void fft(vector<complex<double>>& input)
{
    int n = input.size();

    // ��������
    for (int i = 1, j = 0; i < n; i++)
    {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }

    // ��������
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

// ��FFT����
void ifft(vector<complex<double>>& input)
{
    int n = input.size();

    // ��������
    for (int i = 1, j = 0; i < n; i++)
    {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }

    // ��������
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

    // ��һ��
    for (complex<double>& sample : input)
    {
        sample /= n;
    }
}

vector<vector<complex<double>>> fft2D(const vector<vector<complex<double>>>& input)
{
    int rows = input.size();
    int cols = input[0].size();

    // ���н���һάFFT
    vector<vector<complex<double>>> fftRows(rows);
    for (int i = 0; i < rows; i++)
    {
        fftRows[i] = input[i];
        fft(fftRows[i]);
    }

    // ���н���һάFFT
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

// �źſ�����ؼ���
vector<complex<double>> fastCorrelation(const vector<complex<double>>& signal1, const vector<complex<double>>& signal2)
{
    // �źų���
    int n = signal1.size();
    // ����ʹ�����źų�����ͬ������֤����Ϊ2���ݴ�
    int fftSize = 1;
    while (fftSize < n * 2) {
        fftSize <<= 1;
    }
    // ���źŽ���FFT�任
    vector<complex<double>> fftSignal1(fftSize, 0.0);
    vector<complex<double>> fftSignal2(fftSize, 0.0);

    for (int i = 0; i < n; i++) {
        fftSignal1[i] = signal1[i];
        fftSignal2[i] = signal2[i];
    }
    fft(fftSignal1);
    fft(fftSignal2);
    // ����Ƶ���������źŵĳ˻�
    vector<complex<double>> fftResult(fftSize, 0.0);
    for (int i = 0; i < fftSize; i++) {
        fftResult[i] = fftSignal1[i] * conj(fftSignal2[i]);
    }
    // �Գ˻����������FFT�任
    ifft(fftResult);
    // ������FFT���
    return fftResult;
}


// ��ͨ�������
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


// ���پ��
vector<complex<double>> fastConvolution(const vector<complex<double>>& signal1, const vector<complex<double>>& signal2)
{
    // �źų���
    int n = signal1.size();

    // ����ʹ�����źų�����ͬ������֤����Ϊ2���ݴ�
    int fftSize = 1;
    while (fftSize < n * 2) {
        fftSize <<= 1;
    }

    // ���źŽ���FFT�任
    vector<complex<double>> fftSignal1(fftSize, 0.0);
    vector<complex<double>> fftSignal2(fftSize, 0.0);

    for (int i = 0; i < n; i++) {
        fftSignal1[i] = signal1[i];
        fftSignal2[i] = signal2[i];
    }

    fft(fftSignal1);
    fft(fftSignal2);

    // ����Ƶ���������źŵĳ˻�
    vector<complex<double>> fftResult(fftSize, 0.0);
    for (int i = 0; i < fftSize; i++) {
        fftResult[i] = fftSignal1[i] * fftSignal2[i];
    }

    // �Գ˻����������FFT�任
    ifft(fftResult);

    // ������FFT���
    return fftResult;
}


// �����˷�
vector<int> multiply(const vector<int>& num1, const vector<int>& num2)
{
    int n = num1.size();
    int m = num2.size();
    int fftSize = 1;
    while (fftSize < n + m) fftSize <<= 1;

    vector<complex<double>> fftNum1(fftSize, 0);
    vector<complex<double>> fftNum2(fftSize, 0);

    // ������Ĵ���ת��Ϊ������ʽ
    for (int i = 0; i < n; i++)
    {
        fftNum1[i] = num1[i];
    }
    for (int i = 0; i < m; i++)
    {
        fftNum2[i] = num2[i];
    }

    // ���� FFT �任
    fft(fftNum1);
    fft(fftNum2);

    // ���
    vector<complex<double>> fftResult(fftSize);
    for (int i = 0; i < fftSize; i++)
    {
        fftResult[i] = fftNum1[i] * fftNum2[i];
    }

    // ������ FFT �任
    ifft(fftResult);

    // ��ȡʵ��������������Ϊ�˻����
    vector<int> result(n + m - 1);
    for (int i = 0; i < n + m - 1; i++)
    {
        result[i] = round(fftResult[i].real());
    }

    return result;
}


//int main()
//{
//    vector<int> num1 = { 6,6,6,6 }; // ��һ������
//    vector<int> num2 = { 4, 5, 6 }; // �ڶ�������
//
//    vector<int> product = multiply(num1, num2);
//
//    // ��ӡ�˻����
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
    // ���������ź�
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

    // �����źŵĿ�����ؼ���
    // vector<complex<double>> correlationResult = fastCorrelation(signal1, signal2);

    // ������ؼ��������ļ�
    // saveDataToFile("correlation_result.txt", correlationResult);

    // ��ӡ���
    //cout << "Correlation Result: correlation_result.txt" << endl;
    //for (const complex<double>& sample : correlationResult) {
    //    cout << sample << endl;
    //}

    // �����źŵĿ��پ��
    auto t1 = Clock::now();
    vector<double> normalconvolutionResult = convolution(data, data);
    auto t2 = Clock::now();// ��ʱ����
    cout << "convolution cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";


    t1 = Clock::now();
    vector<complex<double>> convolutionResult = fastConvolution(signal1, signal2);
    t2 = Clock::now();// ��ʱ����
    cout << "fastConvolution cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";
    // ������������ļ�
    saveDataToFile("convolution_result.txt", convolutionResult);

    //vector<complex<double>> num1 = { 1, 2, 3 }; // ��һ������
    //vector<complex<double>> num2 = { 4, 5, 6 }; // �ڶ�������

    //vector<complex<double>> result = multiply(num1, num2);

    //// ������
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
//    // ��ά����
//    vector<vector<complex<double>>> input = {
//        {1, 2, 3, 4},
//        {4, 5, 6, 7},
//        {7, 8, 9, 10},
//        {11,12,13,14}
//    };
//
//    // ���ж�άFFT
//    vector<vector<complex<double>>> result = fft2D(input);
//
//    // ��ӡ���
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