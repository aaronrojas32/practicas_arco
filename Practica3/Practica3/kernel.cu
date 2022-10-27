//Includes
#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <cuda_runtime.h>

//Prototipos de las funciones s
__host__ void propiedades_Device(int deviceID);

//Funcion que calcula los bloques
__host__ int calcula_bloques(int N) {
	int bloques = N / 10;
	if (N % 10 != 0) {
		bloques++;
	}
	return bloques;
}

//Funcion que genera un vector con numeros aleatorios
__host__ void generarVector(int *vector1, int N)
{
	srand((int)time(NULL));
	for (int i = 0; i < N; i++)
	{
		vector1[i] = (int)rand() % 10;
	}
}

//Funcion kernel para generar los vectores
__global__ void kernel(int *vector_1, int *vector_2, int *vector_suma, int N) {
	int idGlobal = threadIdx.x + blockDim.x * blockIdx.x;

	if (idGlobal < N) {
		// creamos el vector 2 (inverso)
		vector_2[idGlobal] = vector_1[(N - 1) - idGlobal];
		// sumamos los dos vectores y escribimos el resultado
		vector_suma[idGlobal] = vector_1[idGlobal] + vector_2[idGlobal];
	}
}

//Funcion main
int main(int argc, char** argv)
{
	// declaracion de variables
	int *hst_vector1, *hst_vector2, *hst_resultado;
	int *dev_vector1, *dev_vector2, *dev_resultado;
	int numeroThreats, numeroBloques;
	int deviceCount;

	//Cargamos los dispostivios cuda y obtenemos sus datos
	cudaGetDeviceCount(&deviceCount);
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

	//Imprimimos los dispostivios y sus propiedades
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("Introduce el numero de elementos: ");
	scanf("%d", &numeroThreats);
	getchar();
	numeroBloques = calcula_bloques(numeroThreats);
	printf("> Vector de %d elementos\n", numeroThreats);
	printf("> Lanzamiento con %d bloques de 10 elementos\n", numeroBloques);

	// reserva de memoria en el host
	hst_vector1 = (int*)malloc(numeroThreats * sizeof(int));
	hst_vector2 = (int*)malloc(numeroThreats * sizeof(int));
	hst_resultado = (int*)malloc(numeroThreats * sizeof(int));

	// reserva de memoria en el device
	cudaMalloc((void**)&dev_vector1, numeroThreats * sizeof(int));
	cudaMalloc((void**)&dev_vector2, numeroThreats * sizeof(int));
	cudaMalloc((void**)&dev_resultado, numeroThreats * sizeof(int));

	// inicializacion del primer vector
	generarVector(hst_vector1, numeroThreats);

	// copiamos el vector 1 en el device
	cudaMemcpy(dev_vector1, hst_vector1, numeroThreats * sizeof(int), cudaMemcpyHostToDevice);

	// inicializacion del segundo vector y suma
	kernel << < numeroBloques, 10 >> >(dev_vector1, dev_vector2, dev_resultado, numeroThreats);

	// recogida de datos desde el device
	cudaMemcpy(hst_vector2, dev_vector2, numeroThreats * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_resultado, dev_resultado, numeroThreats * sizeof(int), cudaMemcpyDeviceToHost);

	// impresion de resultados
	printf("VECTOR 1:\n");
	for (int i = 0; i < numeroThreats; i++)
	{
		printf("%2d ", hst_vector1[i]);
	}
	printf("\n");
	printf("VECTOR 2:\n");
	for (int i = 0; i < numeroThreats; i++)
	{
		printf("%2d ", hst_vector2[i]);
	}
	printf("\n");
	printf("SUMA:\n");
	for (int i = 0; i < numeroThreats; i++)
	{
		printf("%2d ", hst_resultado[i]);
	}
	printf("\n");
	printf("***************************************************\n");
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}

//Funcion para obtener las caracteristicas del dispositivo
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
	int maxBlockSize = deviceProp.maxThreadsPerBlock;
	int hilosX = deviceProp.maxThreadsDim[0];
	int hilosY = deviceProp.maxThreadsDim[1];
	int hilosZ = deviceProp.maxThreadsDim[2];
	int bloquesX = deviceProp.maxGridSize[0];
	int bloquesY = deviceProp.maxGridSize[1];
	int bloquesZ = deviceProp.maxGridSize[2];
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
	printf("> No. maximo de Bloques (por eje): \n");
	printf(" [eje x -> %d] \n", bloquesX);
	printf(" [eje y -> %d] \n", bloquesY);
	printf(" [eje z -> %d] \n", bloquesZ);
	printf("> No. maximo de hilos (por bloque) \t: %d \n", maxBlockSize);
	printf(" [eje x -> %d] \n", hilosX);
	printf(" [eje y -> %d] \n", hilosY);
	printf(" [eje z -> %d] \n", hilosZ);
	printf("***************************************************\n");
}