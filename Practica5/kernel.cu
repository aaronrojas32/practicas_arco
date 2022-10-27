// includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include "device_launch_parameters.h"


// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void reduccion(float *datos, float *suma)
// Funcion que suma los primeros N numeros naturales
{
	// KERNEL con 1 bloque de N hilos
	int N = blockDim.x;
	// indice local de cada hilo
	int myID = threadIdx.x;
	// rellenamos el vector de datos
	datos[myID] = (float)1 / ((myID + 1)*(myID + 1));
	// sincronizamos para evitar riesgos de tipo RAW
	__syncthreads();
	// ******************
	// REDUCCION PARALELA
	// ******************
	int salto = N / 2;
	// realizamos log2(N) iteraciones
	while (salto > 0)
	{
		// en cada paso solo trabajan la mitad de los hilos
		if (myID < salto)
		{
			datos[myID] = datos[myID] + datos[myID + salto];
		}
		// sincronizamos los hilos evitar riesgos de tipo RAW
		__syncthreads();
		salto = salto / 2;
	}
	// ******************
	// Solo el hilo no.'0' escribe el resultado final:
	// evitamos los riesgos estructurales por el acceso a la memoria
	if (myID == 0)
	{
		suma[0] = datos[0];
	}
}

//Funcion para verificar si un numero es potencia de 2
__host__ bool esPotenciaDe2(int n)
{
	//Devuelve true si es potencia de 2
	return (ceil(log2(n)) == floor(log2(n)));
}

// declaracion de funciones
__host__ void propiedades_Device(int deviceID);

int main(int argc, char** argv)
{
	//Buscamos los dispostivos CUDA
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0)
	{
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
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
	//Declaramos las variables que vamos a utilizar
	float *hst_suma;
	float *dev_datos, *dev_suma;
	float  sum, err, raiz;
	int n = 0;
	
	//Pedimos el valor de n y verificamos que sea potencia de 2 y mayor que 0
	do {
		printf("> Introduzca el valor de n. Debe ser potencia de 2 y menor que el numero de hilos por bloque (1024): ");
		scanf("%d", &n);
		getchar();
	} while (n<1024 && n!=0 && !esPotenciaDe2(n));

	// reserva de memoria en el host
	hst_suma = (float*)malloc(1 * sizeof(float));
	// reserva de memoria en el device
	cudaMalloc((void**)&dev_datos, n * sizeof(float));
	cudaMalloc((void**)&dev_suma, 1 * sizeof(float));
	// inicializacion del primer vector

	// LANZAMIENTO DEL KERNEL (1 bloque de n hilos)
	// inicializacion del segundo vector y suma
	//reduccion(int *datos, int *suma)
	reduccion << < 1, n >> >(dev_datos, dev_suma);

	// recogida de datos desde el device
	cudaMemcpy(hst_suma, dev_suma, 1 * sizeof(float), cudaMemcpyDeviceToHost);

	//Cálculo de las soluciones
	//Hacemos la suma de los elementos
	sum = hst_suma[0];
	//Hacemos la raiz cuadrada de la suma y multiplicamos por 6
	raiz = sqrt(sum * 6);
	//Para calcular el error, usamos una regla de 3
	err = (raiz - 3.141593) / 3.141593 * 100;
	err = fabsf(err);

	printf("> RESULTADOS:\n");
	printf("Lanzamiento con 1 bloque de %d hilos\n", n);
	printf("Valor de PI: 3.141593\n");
	printf("Valor que hemos calculado: %f\n", raiz);
	printf("Error relativo: %f%%\n", err);

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
	printf("> MAX Hilos por bloque \t: %d\n", deviceProp.maxThreadsPerBlock);
	printf("> Memoria Global (total) \t: %u MiB\n",
		deviceProp.totalGlobalMem / (1024 * 1024));
	printf("***************************************************\n");
}
/////////////////////////////////////////////////////////////////////////// 