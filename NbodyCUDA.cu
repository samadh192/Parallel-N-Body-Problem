#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "cuda_device_launch_parameters.h"
float dt=0.01;

__global__ void Compute(int *p,int *v,int* m,int *N){
	int id=threadIdx.x,i;
	double Force=0;
	double G=6.674*pow(10,-11);
	for(i=0;i<*N;i++){
		if(i==id)
			continue;
		k=p[id]-p[i];
		Force+=(-1)*G*m[id]*m[i]*(k)/fabs(pow(k,3));
	}
	double a=Force/m[id];
	double newV=v[id]+a*dt;
	double newP=v[id]*t+(1/2)*a*pow(dt,2);
	v[id]=newV;
	p[id]=newP;
}

int main(void){
	int N,i;
	float T;
	printf("Enter number of objects:");
	scanf("%d",&N);
	printf("Enter evaluation time:");
	scanf("%d",&T);
	int steps=int(T/dt);
	double p[N],v[N],m[N];

	printf("Enter the initial position of %d objects:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&p[i]);
	printf("Enter the initial velocity of %d objects:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&v[i]);
	printf("Enter the mass of %d objects:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&m[i]);

	double *d_p,*d_v,*d_m,*d_n;
	int size=N*sizeof(double);

	cudaMalloc((void**)&d_p,size);
	cudaMalloc((void**)&d_v,size);
	cudaMalloc((void**)&d_m,size);
	cudaMalloc((void**)&d_n,sizeof(int));

	cudaMemcpy(d_p,&p,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_v,&v,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_m,&m,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_n,&N,sizeof(int),cudaMemcpyHostToDevice);

	for(i=0;i<steps;i++){
		Compute<<<1,N>>>(d_p,d_v,d_m,d_n);
	}

	cudaMemcpy(&p,d_p,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(&v,d_v,size,cudaMemcpyDeviceToHost);

	printf("\tPosition\tVeocity");
	for(i=0;i<N;i++){
		printf("Object %d:\t%lf\t%lf\n",i+1,p[i],v[i]);
	}

	return 0;
}
