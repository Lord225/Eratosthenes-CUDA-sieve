#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <conio.h>
#include <iostream>
#include <vector>
#include <chrono> 
#include <math.h>
#include <string>

/**********************************************************************************
* Super szybkie równoległe sito erastotelesa v2.0
* Aby skompilować progra trzeba posiadać specjalny kompilator invidii
* Aby uruchomic wspolbierzna wersje algorytmu należy miec karte graficzna invidii
* Opis algorytmu zamieszczony na onedrive
* Algorytm jest w wersji 1.0 (glowne ograniczenie maksymalne N to ok. 1mld
*///*******************************************************************************

#define AutoThreads
//#define Default
//#define DEBUG


#define DefaultSectorNum 16
#define ThreadsPerBlock 512
#define DefaultNum (1<<10)
#define DefaultToShow 1000

#define big unsigned long long int
#define uint unsigned int 
const uint IntSize = (sizeof(int) * 8);

using namespace std;
using namespace std::chrono;

void GetError(cudaError Err) {
	cout << "WYSTAPIL BLAD: " << cudaGetErrorString(Err) << endl;
}

//funkcje GPU ulatwiajace operacje na bitach (dostep do pos'tego bitu w tabeli a)
__device__ inline bool pos(int* a, uint pos) {
	return (a[pos / 32] >> (pos % 32)) & 1;
}
//funkcje GPU ulatwiajace operacje na bitach (ustawienie pos'tego bitu w tabeli a)
__device__ inline void set(int* a, uint pos, bool val = true) {
	if (val) { //ustawia pos'ty bit w ci�gu a na warto�� val
		atomicOr(&a[pos / 32], (1 << (pos % 32)));
	}
	else {
		atomicAnd(&a[pos / 32], ~((int)(1 << (pos % 32))));
	}
}

//Te same funcje ale dla CPU
inline bool Pos(int* a, big pos) {
	return (a[pos / IntSize] >> (pos % IntSize)) & 1;
}
inline void Set(int* a, big pos, bool val) {
	if (val) {
		a[pos / IntSize] |= (1 << (pos % IntSize));
	}
	else {
		a[pos / IntSize] &= ~((char)(1 << (pos % IntSize)));
	}
}

struct Modes {
	big Algoritm;
	big num;
	bool GPU = false;
	bool CPU = false;
	big toShow;
	bool ShowLast;
	bool ShowinRows;
};

//IMPLEMENTACJE ALGORYTMÓW DLA CPU + zoptymalizowana tablica boolowska
class BinaryArry2d
{
	char* Data;
	big AllocSize;
	big Size;
public:
	BinaryArry2d(const big sizeX) {
		big rX = (sizeX - 1) / 8 + 1;
		Data = new char[rX];
		AllocSize = rX;
		Size = sizeX;
	}
	void Fill(bool fill) {
		if (fill) {
			for (int i = 0; i < AllocSize; i++) {
				Data[i] = 255;
			}
		}
		else {
			for (int i = 0; i < AllocSize; i++) {
				Data[i] = 0;
			}
		}
	}
	BinaryArry2d& operator ^= (big x) {
		Data[x / 8] ^= (1 << (x % 8));
		return *this;
	}
	bool operator[](big x) {
		return (Data[x / 8] >> (x % 8)) & 1;
	}

	void set(big x, bool type) {
		if (type) {
			Data[x / 8] |= (1 << (x % 8));
		}
		else {
			Data[x / 8] &= ~((char)(1 << (x % 8)));
		}
	}
	uint size() {
		return Size;
	}
	uint allocSize() {
		return AllocSize;
	}
	~BinaryArry2d() {
		delete[] Data;
	}
};
void SieveOfAtkin(big num, BinaryArry2d &Array) {
	Array.Fill(false);
	Array.set(2, true);
	Array.set(3, true);
	Array.set(5, true);

	big sqrtnum = sqrt(num + 1);
	big n, mod;

	for (big x = 1; x <= sqrtnum; x++)
	{
		for (big y = 1; y <= sqrtnum; y += 2)
		{
			n = 4 * x*x + y * y;
			if (n <= num) {
				mod = n % 60;

				if (mod == 1 || mod == 13 || mod == 17 || mod == 29 || mod == 37 || mod == 41 || mod == 49 || mod == 53) {
					Array ^= n;
				}
			}
		}
	}

	for (big x = 1; x <= sqrtnum; x += 2)
	{
		for (big y = 2; y <= sqrtnum; y += 2)
		{
			n = 3 * x*x + y * y;
			if (n <= num) {
				mod = n % 60;

				if (mod == 7 || mod == 19 || mod == 31 || mod == 43) {
					Array ^= n;
				}
			}
		}
	}

	for (big x = 2; x <= sqrtnum; x++)
	{
		for (big y = 1; y <= x - 1; y++)
		{
			n = 3 * x*x - y * y;

			if (n <= num) {
				mod = n % 60;

				if (mod == 11 || mod == 23 || mod == 47 || mod == 59) {
					Array ^= n;
				}
			}
		}
	}

	for (big i = 5; i <= sqrtnum; i++)
	{
		if (Array[i])
		{
			for (big j = i * i; j <= num; j += i)
			{
				Array.set(j, false);
			}
		}
	}
}
void SieveOfSundaram(big n, BinaryArry2d & Array)
{
	Array.Fill(false);
	const uint nNew = (n - 2) / 2;
	uint pos = 0;
	for (uint i = 1; i <= nNew; i++) {
		for (uint j = i; (i + j + 2 * i*j) <= nNew; j++) {
			pos = i + j + 2 * i*j;
			Array.set(pos, true);
		}
	}

}
void SieveOfEratostenes(big n, BinaryArry2d & Array) {
	uint sqrtn = sqrt(n);
	Array.Fill(true);

	for (uint i = 2; i <= sqrtn; i++) {
		if (Array[i]) {
			for (int j = i * i; j <= n; j += i) {
				Array.set(j, false);
			}
		}
	}
}

//#############################################################
//#															  #
//#                      Funkcje GPU						  #
//#															  #
//#############################################################

//USTAW TABELE a NA SAME FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF...
__global__ void memoryset(int *a, big num, const big AllocSize)
{
	uint tid = blockDim.x*blockIdx.x + threadIdx.x;
	const uint move = blockDim.x * gridDim.x;

	while (tid < AllocSize) {
		a[tid] = -1;
		tid += move;
	}
}

//WSPÓŁBIERZNIE WYTNIJ WSZYSTKIE WIELOKROTNOSCI x Z TABELI a
__global__ void CutMultiples(int *a, const uint xx, const uint x, const uint Num)
{
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	const uint move = blockDim.x * gridDim.x;
	uint Pos = xx + x * tid;

	while (Pos <= Num) {
		set(a, Pos, false);
		tid += move;
		Pos = xx + x * tid;
	}

}
unsigned big Sito_GPUv2(const unsigned big num, vector<uint>&Array, Modes Settings) {
	big BufNum;
#ifndef DefaultSectorNum
	for (uint i = 0; i < 16; i++)
	{
		BufNum = 1 << i;
		if (num / BufNum <= 1073741824) {
			break;
		}
	}
#else
	BufNum = DefaultSectorNum;
#endif // DefaultSectorSize
	auto ts = high_resolution_clock::now();		//chrono - mierzenie czasu 
	const uint sqrtNum = ceil(sqrt(num));
	cudaError Error;							//Wylapywanie bledow CUDA'Y
	cudaStream_t* Streams = new cudaStream_t[BufNum];
	const big AllocSize = (num) / IntSize + 1;	//ROZMIAR ALOKACJI NA CPU
	const big BuffAllocSize = (num / BufNum) / IntSize + 1;//ROZMIAR ALOKACJI SEKTORA
	const big BuffSize = (num / BufNum);//ROZMIAR 1 SEKTORA
	cudaDeviceProp GPU;							//Wlasciwosci GPU

	//Init
	cudaGetDeviceProperties(&GPU, 0);
	for (int i = 0; i < BufNum; i++)
	{
		cudaStreamCreate(&(Streams[i]));
	}

	//Alloc
	int *a = new int[AllocSize];
	int *gpu_a;
	Error = cudaMalloc((void**)&gpu_a, AllocSize * sizeof(int));
	if (Error != cudaSuccess) {
		cout << "Za malo VRAMU" << endl;
		free(a);
		cudaFree(gpu_a);
		return false;
	}

	//Wypelnianie tabeli FFFFFFFFFFFFF
	for (big i = 0; i < AllocSize; i++)
	{
		a[i] = -1;
	}

	//Przygotowywanie siatki
#ifdef AutoThreads
	uint blocks = GPU.multiProcessorCount * 2;
	uint threads = ThreadsPerBlock;
#else
	uint blocks = 256;
	uint threads = 256;
#endif // AutoThreads

	cout << "Specyfikacja GPU 0: " << endl;
	cout << "Nazwa: \t" << GPU.name << endl;
	cout << "Major: \t" << GPU.major << endl;
	cout << "ALOKACJA:" << endl;
	cout << "Sector num: \t" << BufNum << endl;
	cout << "Mem: \t" << GPU.totalGlobalMem << "B, " << GPU.totalGlobalMem*0.000000001f << "GB" << endl;
	cout << "Alloc: \t" << AllocSize * 4 << "B, " << (float)(AllocSize * 4 * 100) / GPU.totalGlobalMem << "%" << endl;
	cout << "SectorSize: \t" << BuffAllocSize * 4 << endl;
	cout << "Sector: \t" << BuffSize << endl;

	////#####################################################////
	////					GŁÓWNY ALGORYTM					 ////
	////#####################################################////

	for (uint x = 0; x < BufNum; x++)
	{
		memoryset << < blocks, threads, 0, Streams[x] >> > ((gpu_a + x * BuffAllocSize), num, BuffAllocSize);
	}
	//Zmienne pomocnicze
	big i = 2, // iterator
		y = 0, // Start
		x = 0,
		z = 0,
		ii = 0, //	kwadrat i
		bii = 0, // (i-1)^2
		size = 0, // ilosc bajtów do skopiowania
		adress = 0; // adres od którego ma zaczac sie kopiowanie



	for (; i <= sqrtNum; i++)//wykonaj na i od 2 do sqrtNum
	{
		if (Pos(a, i)) {	 //Jezeli A[i] == true:
			y = 0;
			ii = i * i;


			x = ii / BuffSize;
			CutMultiples << < blocks, threads, 0, Streams[x] >> > (gpu_a + (ii / IntSize), ii%BuffSize, i, BuffSize);

			x++;
			z = (uint)ceil((float)(BuffSize - i) / i)*i;
			//Wytnij wielokrotnosci i
			for (int* gpu_a_dev = gpu_a + (BuffSize) / IntSize; x < BufNum; x++)
			{
				y = z + i;
				CutMultiples << < blocks, threads, 0, Streams[x] >> > (gpu_a_dev, y%BuffSize, i, BuffSize);

				cudaDeviceSynchronize();
				cudaMemcpy(a, gpu_a, AllocSize * sizeof(int), cudaMemcpyDeviceToHost);

				z = (uint)ceil((float)(y - i) / i)*i;
				gpu_a_dev += (BuffSize) / IntSize;
			}
			//Oblicz adres od ktorego ma zaczac sie kopiowanie i ilosc bajtow do skopiowania
			size = ii / 32 - bii / 32 + 1;
			adress = bii / 32;
			bii = ii;

			cudaDeviceSynchronize();
			//Kopiowanie
			cudaMemcpy(&a[adress], &gpu_a[adress], size * sizeof(int), cudaMemcpyDeviceToHost);
#ifdef DEBUG
			Error = cudaGetLastError();
			if (Error != cudaSuccess) {
				GetError(Error);
				free(a);
				cudaFree(gpu_a);
				return -1;
			}
			int* currentadress = &a[adress];
			int* endadress = &a[AllocSize - 1];
			cudaMemcpy(a, gpu_a, AllocSize * sizeof(int), cudaMemcpyDeviceToHost);
#endif // DEBUG
		}
	}

	//Ostatnia iteracja nie kopjuje wiec skopjuj od (i-1)^2 do konca tabeli
	i -= 1;
	ii = i * i;
	size = ii / 32 - bii / 32 + 1;
	i -= 1;
	bii = i * i;
	adress = AllocSize - size;

	cudaMemcpy(&a[adress], &gpu_a[adress], size * sizeof(int), cudaMemcpyDeviceToHost);

	//Zmierz czas
	auto te = high_resolution_clock::now();
	auto GPU_Time = duration_cast<milliseconds>(te - ts);

	//Wpisz wyniki do tablicy dynamicznej
	Array.clear();
	uint counter = 0;
	for (big i = 2; i < num; i++)
	{
		if (Pos(a, i)) {
			Array.push_back(i);
		}
		if (counter > Settings.toShow) {
			break;
		}
		counter++;
	}

	if (Settings.ShowLast) {
		for (big i = Settings.num - 1; i > 0; i--)
		{
			if (Pos(a, i)) {
				Array.push_back(i);
				break;
			}
		}
	}

	free(a);
	cudaFree(gpu_a);
	return GPU_Time.count();
}//NIEDZIALA
unsigned big Sito_GPU(const unsigned big num, vector<uint>&Array, Modes Settings) {

	auto ts = high_resolution_clock::now();		//chrono - mierzenie czasu 
	const uint sqrtNum = ceil(sqrt(num));
	cudaError Error;							//Wylapywanie bledow CUDA'Y
	cudaStream_t Stream;						//Bajer do optymalizacji
	const big AllocSize = (num) / IntSize + 1;	//Rozmiar Alokacji
	cudaDeviceProp GPU;							//Wlasciwosci GPU

	//Init
	cudaGetDeviceProperties(&GPU, 0);
	cudaStreamCreate(&Stream);

	//Alloc
	int *a = new int[AllocSize];
	int *gpu_a;
	Error = cudaMalloc((void**)&gpu_a, AllocSize * sizeof(int));
	if (Error != cudaSuccess) {
		cout << "Za malo VRAMU" << endl;
		free(a);
		cudaFree(gpu_a);
		return false;
	}

	//Wypelnianie tabeli FFFFFFFFFFFFF
	for (big i = 0; i < AllocSize; i++)
	{
		a[i] = -1;
	}

	//Przygotowywanie siatki
#ifdef AutoThreads
	uint blocks = GPU.multiProcessorCount * 2;
	uint threads = ThreadsPerBlock;
#else
	uint blocks = 256;
	uint threads = 256;
#endif // AutoThreads

	cout << "Specyfikacja GPU 0: " << endl;
	cout << "Nazwa: \t" << GPU.name << endl;
	cout << "Major: \t" << GPU.major << endl;
	cout << "Mem: \t" << GPU.totalGlobalMem << "B, " << GPU.totalGlobalMem*0.000000001f << "GB" << endl;
	cout << "Alloc: \t" << AllocSize * 4 << "B, " << (float)(AllocSize * 4 * 100) / GPU.totalGlobalMem << "%" << endl;

	////#####################################################////
	////					GŁÓWNY ALGORYTM					 ////
	////#####################################################////

	memoryset << < blocks, threads >> > (gpu_a, num, AllocSize);

	//Zmienne pomocnicze
	big i = 2, // iterator
		ii = 0, //	kwadrat i
		bii = 0, // (i-1)^2
		size = 0, // ilosc bajtów do skopiowania
		adress = 0; // adres od którego ma zaczac sie kopiowanie

	for (; i <= sqrtNum; i++)//wykonaj na i od 2 do sqrtNum
	{
		if (Pos(a, i)) {	 //Jezeli A[i] == true:
			ii = i * i;

			//Wytnij wielokrotnosci i
			CutMultiples << < blocks, threads, 0, Stream >> > (gpu_a, ii, i, num);

			//Oblicz adres od ktorego ma zaczac sie kopiowanie i ilosc bajtow do skopiowania
			size = ii / 32 - bii / 32 + 1;
			adress = bii / 32;
			bii = ii;

#ifdef DEBUG
			Error = cudaGetLastError();
			if (Error != cudaSuccess) {
				GetError(Error);
				free(a);
				cudaFree(gpu_a);
				return -1;
			}
			int* currentadress = &a[adress];
			int* endadress = &a[AllocSize - 1];
#endif // DEBUG

			//Kopiowanie
			cudaMemcpyAsync(&a[adress], &gpu_a[adress], size * sizeof(int), cudaMemcpyDeviceToHost, Stream);
		}
	}

	//Ostatnia iteracja nie kopjuje wiec skopjuj od (i-1)^2 do konca tabeli
	i -= 1;
	ii = i * i;
	size = ii / 32 - bii / 32 + 1;
	i -= 1;
	bii = i * i;
	adress = AllocSize - size;

	cudaMemcpy(&a[adress], &gpu_a[adress], size * sizeof(int), cudaMemcpyDeviceToHost);

	//Zmierz czas
	auto te = high_resolution_clock::now();
	auto GPU_Time = duration_cast<milliseconds>(te - ts);

	//Wpisz wyniki do tablicy dynamicznej
	Array.clear();
	uint counter = 0;
	for (big i = 2; i < num; i++)
	{
		if (Pos(a, i)) {
			Array.push_back(i);
		}
		if (counter > Settings.toShow) {
			break;
		}
		counter++;
	}

	if (Settings.ShowLast) {
		for (big i = Settings.num - 1; i > 0; i--)
		{
			if (Pos(a, i)) {
				Array.push_back(i);
				break;
			}
		}
	}

	free(a);
	cudaFree(gpu_a);
	return GPU_Time.count();
}
unsigned big Sito_CPU(const big num, vector<uint>&Array, Modes Settings) {
	auto ts = high_resolution_clock::now();

	//Tablica boolowska
	BinaryArry2d binArray(num + 1);

	//Wybierz algorytm
	switch (Settings.Algoritm)
	{
	case 1:
		SieveOfEratostenes(num, binArray);
		break;
	case 2:
		SieveOfAtkin(num, binArray);
		break;
	case 3:
		SieveOfSundaram(num, binArray);
		break;
	default:
		break;
	}

	//Zmierz czas
	auto te = high_resolution_clock::now();
	auto CPU_Time = duration_cast<milliseconds>(te - ts);

	//Wpisz wyniki do tabeli dynamicznej
	Array.clear();
	uint counter = 0;
	if (Settings.Algoritm != 3) {
		for (uint i = 2; i < num; i++)
		{
			if (binArray[i]) {
				Array.push_back(i);
			}
			if (counter > Settings.toShow) {
				break;
			}
			counter++;
		}
		if (Settings.ShowLast) {
			for (int i = Settings.num; i > 0; i--)
			{
				if (binArray[i]) {
					Array.push_back(i);
					break;
				}
			}
		}
	}
	else {
		Array.push_back(2);
		for (uint i = 1; i < (num - 2) / 2; i++)
		{
			if (!binArray[i]) {
				Array.push_back(2 * i + 1);
			}
			if (counter > Settings.toShow) {
				break;
			}
			counter++;
		}
		if (Settings.ShowLast) {
			for (big i = (num - 2) / 2; i > 0; i--)
			{
				if (!binArray[i]) {
					Array.push_back(2 * i + 1);
					break;
				}
			}
		}
	}
	return (big)CPU_Time.count();
}

Modes Load() {
	cout << "******************************************" << endl;
	cout << "*********SUPER SPEED SIEVE ON GPU*********" << endl;
	cout << "*************M. Zlotorowicz***************" << endl;
	cout << "******************************************" << endl;

	cout << endl;
#ifndef Default
	cout << "Ktorego urzadzenia chcesz uzyc?" << endl;
	cout << "1. GPU" << endl;
	cout << "2. CPU" << endl;
	cout << "3. GPU i CPU" << endl;

	Modes settings;

	int dev;
	do {
		cin >> dev;
	} while (dev > 3);

	switch (dev)
	{
	case 1:
		settings.GPU = true;
		break;
	case 2:
		settings.CPU = true;
		break;
	case 3:
		settings.CPU = true;
		settings.GPU = true;
		break;
	default:
		break;
	}

	if (settings.CPU) {
		cout << "Jakiego Sita chcesz uzyc (CPU)?" << endl;
		cout << "1. Erastotenesa" << endl;
		cout << "2. Atkina" << endl;
		cout << "3. Sundarama" << endl;
		do {
			cin >> settings.Algoritm;
		} while (settings.Algoritm > 3);
	}
	else {
		settings.Algoritm = -1;
	}

	string NUM;
	cout << "Podaj zakres do sprawdzenia(MAX: 2147483648 (31T):" << endl;
	cout << "LiczbaGB - Tyle liczb by zmiescily sie w podanej liczbe GB" << endl;
	cout << "LiczbaZ - 1 i Z zer" << endl;
	cout << "LiczbaT - 2^T potegi" << endl;

	cin >> NUM;
	size_t at = NUM.find("GB", 0);
	if (at != string::npos) {
		settings.num = stoi(NUM.substr(0, NUM.size() - 2)) * 8000000000;
	}
	else {
		size_t at = NUM.find("Z", 0);
		if (at != string::npos) {
			settings.num = 1 * pow(10, stoi(NUM.substr(0, NUM.size() - 1)));
		}
		else {
			size_t at = NUM.find("T", 0);
			if (at != string::npos) {
				settings.num = pow(2, stoi(NUM.substr(0, NUM.size() - 1)));
			}
			else {
				settings.num = stoll(NUM);
			}

		}
	}
	cout << settings.num << " Liczb" << endl;

	if (settings.num > 60000000000 && settings.CPU) {
		cout << "UWAGA! dla tak duzego n CPU moze liczyc naprawde dlugi czas..." << endl;
	}

	cout << "Podaj ilosc Elementow do wyswietlenia" << endl;
	cin >> settings.toShow;

	cout << "Czy chcesz zobaczyc Ostatnia liczbe pierwsza? (t/n)" << endl;
	char t;
	cin >> t;
	if (t == 't') {
		settings.ShowLast = true;
	}
	else {
		settings.ShowLast = false;
	}

	cout << "Wyswietlic w rzedach? (t/n)" << endl;
	cin >> t;
	if (t == 't') {
		settings.ShowinRows = true;
	}
	else {
		settings.ShowinRows = false;
	}
#else //Default
	Modes settings;
	settings.Algoritm = 2;
	settings.CPU = true;
	settings.GPU = true;
	settings.num = DefaultNum;
	settings.ShowLast = true;
	settings.toShow = DefaultToShow;
	settings.ShowinRows = true;
#endif
	return settings;
}
void Show(vector<uint>primesGPU, vector<uint>primesCPU, Modes settings, big CPU_time, big GPU_time) {
	if (settings.GPU) {
		cout << "GPU";
	}
	if (settings.CPU) {
		cout << "\t" << "CPU";
	}

	cout << endl;
	if (settings.GPU && settings.CPU) {
		cout << "GPU JEST SZYBSZE OD CPU      " << (float)CPU_time / GPU_time << " razy. " << endl;
	}
	if (settings.GPU) {
		cout << GPU_time << "ms";
	}
	if (settings.CPU) {
		cout << "\t" << CPU_time << "ms";
	}

	cout << endl;

	if (!settings.ShowinRows) {
		if (settings.GPU) {
			cout << "GPU: [";
			for (auto a : primesGPU) {
				cout << a << ", ";
			}
			cout << "]";
		}
		if (settings.CPU) {
			cout << "CPU: [";
			for (auto a : primesCPU) {
				cout << a << ", ";
			}
			cout << "]";
		}

	}
	else {
		auto i = primesGPU.begin(), j = primesCPU.begin();
		for (;;)
		{
			if (settings.GPU) {
				cout << *i;
			}
			if (settings.CPU) {
				cout << "\t" << *j;
			}

			if (i != primesGPU.end()) {
				i++;
			}
			if (j != primesCPU.end()) {
				j++;
			}

			if (i == primesGPU.end() || j == primesCPU.end()) {
				break;
			}
			cout << endl;
		}
	}
	cout << endl;
}

int main(void)
{
	vector<uint>primesGPU;
	vector<uint>primesCPU;
	big CPU_time = -1, GPU_time = -1;


	Modes settings = Load();

	cout << endl;
	cout << endl;

	if (settings.GPU) {
		cout << "GPU Rozpoczyna Prace" << endl;
		GPU_time = Sito_GPU(settings.num, primesGPU, settings);
		cout << "GPU Zakonczylo Prace" << endl;
	}

	if (settings.CPU) {
		cout << "CPU Rozpoczyna Prace" << endl;
		CPU_time = Sito_CPU(settings.num, primesCPU, settings);
		cout << "CPU Zakonczylo Prace" << endl;
	}


	Show(primesGPU, primesCPU, settings, CPU_time, GPU_time);
	system("pause");
	return 0;
}