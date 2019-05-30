#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Compute(double *px,double *vx,double *py,double *vy,double *pz,double *vz,double* m,int *N){
	double dt=0.01;
	double multiplier=1000000;
	int id=threadIdx.x;
	double Forcex=0;
	double Forcey=0;
	double Forcez=0;
	double G=6.674*0.001;
	//double G=10;
	for(int i=0;i<*N;i++){
		if(i==id)
			continue;
		double kx=px[id]-px[i];
		double k1x;
		double ky=py[id]-py[i];
		double k1y;
		double kz=pz[id]-pz[i];
		double k1z;
		//printf("K[%d]=%lf\n",id,k);
		if(kx<0)
			k1x=-kx;
		else
			k1x=kx;
		if(ky<0)
			k1y=-ky;
		else
			k1y=ky;
		if(kz<0)
			k1z=-kz;
		else
			k1z=kz;
		//printf("K1[%d]=%lf\n",id,k1);
		Forcex+=(-1)*G*m[id]*m[i]*(kx)/(k1x*k1x*k1x);
		Forcey+=(-1)*G*m[id]*m[i]*(ky)/(k1y*k1y*k1y);
		Forcez+=(-1)*G*m[id]*m[i]*(kz)/(k1z*k1z*k1z);
	}
	double ax=(Forcex/m[id]);
	double newVx=vx[id]+ax*dt;
	double changePx=vx[id]*dt+(0.5)*ax*(dt*dt);
	double newPx=px[id]+changePx;
	px[id]=newPx;
	double ay=(Forcey/m[id]);
	double newVy=vy[id]+ay*dt;
	double changePy=vy[id]*dt+(0.5)*ay*(dt*dt);
	double newPy=py[id]+changePy;
	py[id]=newPy;
	double az=(Forcez/m[id]);
	double newVz=vz[id]+az*dt;
	double changePz=vz[id]*dt+(0.5)*az*(dt*dt);
	double newPz=pz[id]+changePz;
	pz[id]=newPz;
	printf("Forcex[%d]:%lf, Accx[%d]:%lf, NewVx[%d]:%lf, NewPx[%d]:%lf\n",id,Forcex,id,ax,id,newVx,id,newPx);
	printf("Forcey[%d]:%lf, Accy[%d]:%lf, NewVy[%d]:%lf, NewPy[%d]:%lf\n",id,Forcey,id,ay,id,newVy,id,newPy);
	printf("Forcez[%d]:%lf, Accz[%d]:%lf, NewVz[%d]:%lf, NewPz[%d]:%lf\n",id,Forcez,id,az,id,newVz,id,newPz);
	vx[id]=newVx;
	vy[id]=newVy;
	vz[id]=newVz;
	//p[id]+=newP;
}

int main(void){
	int N,i;
	double T;
	double dt=0.01;
	printf("Enter number of objects:");
	scanf("%d",&N);
	printf("Enter evaluation time:");
	scanf("%lf",&T);
	int steps=int(T/dt);
	printf("Steps :%d\n",steps);
	double px[N],vx[N],m[N],py[N],vy[N],pz[N],vz[N];

	printf("\nEnter the initial position of %d objects in x direction:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&px[i]);
	printf("Enter the initial position of %d objects in y direction:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&py[i]);
	printf("Enter the initial position of %d objects in z direction:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&pz[i]);
	printf("Enter the initial velocity of %d objects in x direction:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&vx[i]);
	printf("Enter the initial velocity of %d objects in y direction:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&vy[i]);
	printf("Enter the initial velocity of %d objects in z direction:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&vz[i]);
	printf("Enter the mass of %d objects:\n",N);
	for(i=0;i<N;i++)
		scanf("%lf",&m[i]);

	double *d_px,*d_vx,*d_m,*d_py,*d_vy,*d_pz,*d_vz;
	int *d_n;
	int size=N*sizeof(double);

	cudaMalloc((void**)&d_px,size);
	cudaMalloc((void**)&d_vx,size);
	cudaMalloc((void**)&d_py,size);
	cudaMalloc((void**)&d_vy,size);
	cudaMalloc((void**)&d_pz,size);
	cudaMalloc((void**)&d_vz,size);
	cudaMalloc((void**)&d_m,size);
	cudaMalloc((void**)&d_n,sizeof(int));

	cudaMemcpy(d_px,&px,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_vx,&vx,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_py,&py,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy,&vy,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_pz,&pz,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_vz,&vz,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_m,&m,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_n,&N,sizeof(int),cudaMemcpyHostToDevice);

	/*printf("Position: ");
		for(i=0;i<N;i++){
			printf("%lf ",p[i]);
		}
	printf("\nVelocity: ");
		for(i=0;i<N;i++){
			printf("%lf ",v[i]);
		}
	printf("\n");
	 */
	for(i=0;i<steps;i++){
		Compute<<<1,N>>>(d_px,d_vx,d_py,d_vy,d_pz,d_vz,d_m,d_n);
	}

	cudaMemcpy(&px,d_px,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(&vx,d_vx,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(&py,d_py,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(&vy,d_vy,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(&pz,d_pz,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(&vz,d_vz,size,cudaMemcpyDeviceToHost);

	printf("\nPosition in x direction: ");
	for(i=0;i<N;i++){
		printf("%lf ",px[i]);
	}
	printf("\n");
	printf("\nPosition in y direction: ");
	for(i=0;i<N;i++){
		printf("%lf ",py[i]);
	}
	printf("\n");
	printf("\nPosition in z direction: ");
	for(i=0;i<N;i++){
		printf("%lf ",pz[i]);
	}
	printf("\n");
	printf("\nVelocity in x direction: ");
	for(i=0;i<N;i++){
		printf("%lf ",vx[i]);
	}
	printf("\n");
	printf("\nVelocity in y direction: ");
	for(i=0;i<N;i++){
		printf("%lf ",vy[i]);
	}
	printf("\n");
	printf("\nVelocity in z direction: ");
	for(i=0;i<N;i++){
		printf("%lf ",vz[i]);
	}
	printf("\n");
	return 0;
}
