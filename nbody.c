#include "mpi.h"
#include <stdio.h>

void main(int argc,char *argv[]){
	int rank,size;
	int N,i;
	double T;
	double dt=0.01;
	int steps;
	double px[10],vx[10],m[10];
	double py[10],vy[10];
	double pz[10],vz[10];
	double start,end;
	MPI_Init(&argc,&argv);
	start= MPI_Wtime();
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
		
	if(rank==0){
		printf("Enter number of objects:");
		fflush(stdout);
		scanf("%d",&N);

		printf("Enter evaluation time:");
		fflush(stdout);
		scanf("%lf",&T);

		printf("Enter the initial position of %d objects in x-direction:\n",N);
		fflush(stdout);
		for(int i=0;i<N;i++){
			scanf("%lf",&px[i]);
		}

		printf("Enter the initial position of %d objects in y-direction:\n",N);
		fflush(stdout);
		for(int i=0;i<N;i++){
			scanf("%lf",&py[i]);
		}

		printf("Enter the initial position of %d objects in z-direction:\n",N);
		fflush(stdout);
		for(int i=0;i<N;i++){
			scanf("%lf",&pz[i]);
		}
			
		printf("Enter the initial velocity of %d objects in x-direction:\n",N);
		fflush(stdout);
		for(i=0;i<N;i++){
			scanf("%lf",&vx[i]);
		}

		printf("Enter the initial velocity of %d objects in y-direction:\n",N);
		fflush(stdout);
		for(i=0;i<N;i++){
			scanf("%lf",&vy[i]);
		}

		printf("Enter the initial velocity of %d objects in z-direction:\n",N);
		fflush(stdout);
		for(i=0;i<N;i++){
			scanf("%lf",&vz[i]);
		}
			
		printf("Enter the mass of %d objects:\n",N);
		fflush(stdout);
		for(i=0;i<N;i++){
			scanf("%lf",&m[i]);
		}
			//printf("Hello from %d\n",rank);
	}
		//printf("Hello from %d\n",rank);
	MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&T,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&px,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&vx,N,MPI_DOUBLE,0,MPI_COMM_WORLD);

	MPI_Bcast(&py,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&vy,N,MPI_DOUBLE,0,MPI_COMM_WORLD);

	MPI_Bcast(&pz,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&vz,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&m,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
		/*if(rank==1){
			printf("N:%d\n",N);
			printf("P:");
			for(int i=0;i<N;i++){
				printf("%lf ",p[i]);
			}
			printf("\n");
		}*/

	//}
	steps=(int)(T/dt);
	double FinalPx,FinalPy,FinalPz,FinalVx,FinalVy,FinalVz;
	//printf("%d\n",steps);
	//printf("1\n");
	for(i=0;i<steps;i++){
		//printf("2\n");
		double Forcex=0;
		double G=6.674*0.001;
		for(int i=0;i<N;i++){
			if(i==rank)
				continue;
			double kx=px[rank]-px[i];
			double k1x;
		//printf("K[%d]=%lf\n",id,k);
			if(kx<0)
				k1x=-kx;
			else
				k1x=kx;
		//printf("K1[%d]=%lf\n",id,k1);
			Forcex+=(-1)*G*m[rank]*m[i]*(kx)/(k1x*k1x*k1x);
		}
		double ax=(Forcex/m[rank]);
		double newVx=vx[rank]+ax*dt;
		double changePx=vx[rank]*dt+(0.5)*ax*(dt*dt);
		double newPx=px[rank]+changePx;
		px[rank]=newPx;
		printf("Forcex[%d]:%lf,Accx[%d]:%lf,NewVx[%d]:%lf,NewPx[%d]:%lf\n",rank,Forcex,rank,ax,rank,newVx,rank,newPx);
		vx[rank]=newVx;
		MPI_Bcast(&px[rank],1,MPI_DOUBLE,rank,MPI_COMM_WORLD);
		MPI_Bcast(&vx[rank],1,MPI_DOUBLE,rank,MPI_COMM_WORLD);

		double Forcey=0;
		for(int i=0;i<N;i++){
			if(i==rank)
				continue;
			double ky=py[rank]-py[i];
			double k1y;
		//printf("K[%d]=%lf\n",id,k);
			if(ky<0)
				k1y=-ky;
			else
				k1y=ky;
		//printf("K1[%d]=%lf\n",id,k1);
			Forcey+=(-1)*G*m[rank]*m[i]*(ky)/(k1y*k1y*k1y);
		}
		double ay=(Forcey/m[rank]);
		double newVy=vy[rank]+ay*dt;
		double changePy=vy[rank]*dt+(0.5)*ay*(dt*dt);
		double newPy=py[rank]+changePy;
		py[rank]=newPy;
		printf("Forcey[%d]:%lf,Accy[%d]:%lf,NewVy[%d]:%lf,NewPy[%d]:%lf\n",rank,Forcey,rank,ay,rank,newVy,rank,newPy);
		vy[rank]=newVy;
		MPI_Bcast(&py[rank],1,MPI_DOUBLE,rank,MPI_COMM_WORLD);
		MPI_Bcast(&vy[rank],1,MPI_DOUBLE,rank,MPI_COMM_WORLD);

		double Forcez=0;
		for(int i=0;i<N;i++){
			if(i==rank)
				continue;
			double kz=pz[rank]-pz[i];
			double k1z;
		//printf("K[%d]=%lf\n",id,k);
			if(kz<0)
				k1z=-kz;
			else
				k1z=kz;
		//printf("K1[%d]=%lf\n",id,k1);
			Forcez+=(-1)*G*m[rank]*m[i]*(kz)/(k1z*k1z*k1z);
		}
		double az=(Forcez/m[rank]);
		double newVz=vz[rank]+az*dt;
		double changePz=vz[rank]*dt+(0.5)*az*(dt*dt);
		double newPz=pz[rank]+changePz;
		pz[rank]=newPz;
		printf("Forcez[%d]:%lf,Accz[%d]:%lf,NewVz[%d]:%lf,NewPz[%d]:%lf\n",rank,Forcez,rank,az,rank,newVz,rank,newPz);
		vz[rank]=newVz;
		MPI_Bcast(&pz[rank],1,MPI_DOUBLE,rank,MPI_COMM_WORLD);
		MPI_Bcast(&vz[rank],1,MPI_DOUBLE,rank,MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	end=MPI_Wtime();
	if(rank==0){
		printf("\nTime taken:%lf",end-start);
	}
	MPI_Finalize();
}