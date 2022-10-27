// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__host__ void propiedades_Device(int deviceID);
// declaracion de funciones
__host__ void impares_CPU(int *hst_impares, int N)
{
	for (int i = 0; i < N; i++)
	{
		hst_impares[i] = (int)rand() % 10;
	}
}
__global__ void suma_GPU(int *vector_1, int *vector_2, int *vector_suma, int N)
{
	// KERNEL de 1 BLOQUE
	// identificador de hilo
	int id = threadIdx.x;
	// generamos el vector 2
	vector_2[id] = vector_1[(N - 1) - id];
	// sumamos los dos vectores y escribimos el resultado
	vector_suma[id] = vector_1[id] + vector_2[id];
}
int main(int argc, char** argv)
{
	// declaracion de variables
	int *hst_vector1, *hst_vector2, *hst_resultado;
	int *dev_vector1, *dev_vector2, *dev_resultado;
	int N, deviceID;
	unsigned int p;
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cudaDeviceProp deviceProp;
	if (deviceCount == 0)
	{
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		return 1;
	}
	else
	{
		printf("Se han encontrado <%d> dispositivos CUDA:\n", deviceCount);
		for (int id = 0; id < deviceCount; id++)
		{
			propiedades_Device(id);
		}
	}
	// salida del programa
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&deviceProp, deviceID);
	p = deviceProp.maxThreadsPerBlock;
	do {
		printf("Introduce el numero de elementos: ");
		scanf("%d", &N);
		getchar();
		if (N > p || N < 0) {
			printf("> ERROR: numero maximo de hilos superado! [%d hilos]\n", p);
		}
	} while (N > p || N < 0);
	printf("> Vector de %d elementos\n", N);
	printf("> Lanzamiento con 1 bloque de %d elementos\n", N);
	// reserva de memoria en el host
	hst_vector1 = (int*)malloc(N * sizeof(int));
	hst_vector2 = (int*)malloc(N * sizeof(int));
	hst_resultado = (int*)malloc(N * sizeof(int));
	// reserva de memoria en el device
	cudaMalloc((void**)&dev_vector1, N * sizeof(int));
	cudaMalloc((void**)&dev_vector2, N * sizeof(int));
	cudaMalloc((void**)&dev_resultado, N * sizeof(int));
	// inicializacion del primer vector
	impares_CPU(hst_vector1, N);
	// copiamos el vector 1 en el device
	cudaMemcpy(dev_vector1, hst_vector1, N * sizeof(int), cudaMemcpyHostToDevice);
	// LANZAMIENTO DEL KERNEL (1 bloque de N hilos)
	// inicializacion del segundo vector y suma
	suma_GPU << < 1, N >> >(dev_vector1, dev_vector2, dev_resultado, N);
	// recogida de datos desde el device
	cudaMemcpy(hst_vector2, dev_vector2, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_resultado, dev_resultado, N * sizeof(int), cudaMemcpyDeviceToHost);
	// impresion de resultados
	printf("VECTOR 1:\n");
	for (int i = 0; i < N; i++)
	{
		printf("%2d ", hst_vector1[i]);
	}
	printf("\n");
	printf("VECTOR 2:\n");
	for (int i = 0; i < N; i++)
	{
		printf("%2d ", hst_vector2[i]);
	}
	printf("\n");
	printf("SUMA:\n");
	for (int i = 0; i < N; i++)
	{
		printf("%2d ", hst_resultado[i]);
	}
	printf("\n");
	// salida
	printf("***************************************************\n");
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}
__host__ void propiedades_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	// calculo del numero de cores (SP)
	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	const char *archName;
	switch (major)
	{
	case 1:
		//TESLA
		archName = "TESLA";
		cudaCores = 8;
		break;
	case 2:
		//FERMI
		archName = "FERMI";
		if (minor == 0)
			cudaCores = 32;
		else
			cudaCores = 48;
		break;
	case 3:
		//KEPLER
		archName = "KEPLER";
		cudaCores = 192;
		break;
	case 5:
		//MAXWELL
		archName = "MAXWELL";
		cudaCores = 128;
		break;
	case 6:
		//PASCAL
		archName = "PASCAL";
		cudaCores = 64;
		break;
	case 7:
		//VOLTA(7.0) //TURING(7.5)
		cudaCores = 64;
		if (minor == 0)
			archName = "VOLTA";
		else
			archName = "TURING";
		break;
	case 8:
		// AMPERE
		archName = "AMPERE";
		cudaCores = 64;
		break;
	default:
		//ARQUITECTURA DESCONOCIDA
		archName = "DESCONOCIDA";
	}
	int rtV;
	cudaRuntimeGetVersion(&rtV);
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> CUDA Toolkit \t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> Arquitectura CUDA \t: %s\n", archName);
	printf("> Capacidad de Computo \t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores \t: %d\n", SM);
	printf("> No. Nucleos CUDA (%dx%d) \t: %d\n", cudaCores, SM, cudaCores*SM);
	printf("> Memoria Global (total) \t: %u MiB\n",
		deviceProp.totalGlobalMem / (1024 * 1024));
	printf("***************************************************\n");
}