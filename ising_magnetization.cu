#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define L 8 	// lattice length; must be even number!
#define D 3 	// dimensions
#define N 512   // number of spins on square lattice (N=L^D)
#define J_IN 	25.00
#define J_OU 	100.00
#define J_PB   -300.00
#define N_SWEEPS	100000
#define H_max 200
#define H_min -200

/* Distribute N/2 spins over threads and blocks:
   since we do "black" and "white" spins seperately, we only need half the threads:
   N_BLOCKS * N_THREADS_PER_BLOCK = N/2 */
#define N_BLOCKS		8   // number of blocks
#define N_THREADS_PER_BLOCK	32  // number of threads per block


/*
Generator of random numbers Tausworthe in the device:
	Three-step generator with period 2^88.

Keyword Arguments:
z1: unsigned (first step storage, is a random numbert)
z2: unsigned (second step storage, is a random numbert)
z3: unsigned (third step storage, is a random numbert)
*/
__device__ unsigned Tausworthe88(unsigned &z1, unsigned &z2, unsigned &z3)
{
	unsigned b = (((z1 << 13) ^ z1) >> 19);
	z1 = (((z1 & 4294967294) << 12) ^ b);

	b = (((z2 << 2) ^ z2) >> 25);
	z2 = (((z2 & 4294967288) << 4)  ^ b);

	b = (((z3 << 3) ^ z3) >> 11);
	z3 = (((z3 & 4294967280) << 17) ^ b);

	return z1 ^ z2 ^ z3;
}


/*
Generator of random numbers LCRNG in the device:
	Linear congruential random number generators with period 2^32.

Keyword Arguments:
z: unsigned (is a random numbert)
*/
__device__ unsigned LCRNG(unsigned &z)  
{  
	const unsigned a = 1664525, c = 1013904223;
	return z = a * z + c;
}


/*
Combination of a Tausworthe generator with a LCRNG, esulting in a 
generator with a period of about 2^120.

Keyword Arguments:
z1: unsigned (is a random numbert)
z2: unsigned (is a random numbert)
z3: unsigned (is a random numbert)
z: unsigned (is a random numbert)
*/
__device__ float TauswortheLCRNG(unsigned &z1, unsigned &z2, unsigned &z3, unsigned &z)
{
	// combine both generators and normalize 0...2^32 to 0...1
	return (Tausworthe88(z1, z2, z3) ^ LCRNG(z)) * 2.3283064365e-10;
}



/* 
Metropolis algorithm in the device. For this a decomposition was used in chessboard.

Keyword Arguments:
seeds: int * (random number generator seeds)
spins: int * (spin list)
neighbors: unsigned * (list of neighbors of a given spin)
spinIdList: unsigned * (PONER QUE ES ESTO)
*/

__global__ void runMetropolis(int *seeds, int* spins, unsigned* neighbors, unsigned* spinIdList, 
						int* energyDifferences, int* magnetization, bool* contrt, float beta, int H)
{
	const unsigned id = blockDim.x*blockIdx.x + threadIdx.x;

	// spin id from list of black or white spins:
	const unsigned spinId = spinIdList[id];

	// get seed values:
	unsigned z1 = seeds[4*id    ];	// Tausworthe seeds
	unsigned z2 = seeds[4*id + 1];
	unsigned z3 = seeds[4*id + 2];
	unsigned z  = seeds[4*id + 3];	// LCRNG seed

	// energy differences for this block:
	__shared__ int deltaE[N_THREADS_PER_BLOCK];
	__shared__ int blockSpins[N_THREADS_PER_BLOCK];
	deltaE[threadIdx.x] = 0;

	int spinstate = spins[spinId];	// get spin state from DRAM
	int nb[2*D];						// neighbor states

	bool conex = contrt[spinId];
    bool nl[2 * D];
    
	
	// get neighbor states:
	for(unsigned n=0; n<2*D; n++){
		nb[n] = spins[neighbors[2 * D * spinId + n]];
		nl[n] = contrt[neighbors[2 * D * spinId + n]];
        
	}

	// propose random new spin state:
	int newstate = - spinstate;

	// energy difference: E'-E
	int E_before = 0;
	int E_after  = 0;

	for(int n=0; n<2*D; n++){
		
		if(n != 4 && n != 5)
			E_before += J_IN * spinstate * nb[n];
		else{

			if (n == 4 && conex)
				E_before += J_PB * spinstate * nb[n];
			else if (n == 5 && nl[5])
				E_before += J_PB * spinstate * nb[n];
			else
				E_before += J_OU * spinstate * nb[n];
		}

		if(n != 4 && n != 5)
			E_after += J_IN * newstate * nb[n];
		else{

			if (n == 4 && conex)
				E_after += J_PB * newstate * nb[n];
			else if (n == 5 && nl[5])
				E_before += J_PB * newstate * nb[n];
			else
				E_after += J_OU * newstate * nb[n];
		}
	}

	E_before += H * spinstate;
	E_after  += H * newstate;

	// acceptance probability:
	float dE = __int2float_rn(E_before - E_after);
	float pAccept = __expf(-beta*dE);

	if(TauswortheLCRNG(z1, z2, z3, z) <= pAccept)
	{
		spins[spinId] = newstate;   // flip spin
		spinstate = newstate;

		deltaE[threadIdx.x] = E_before - E_after;	// note energy difference
	}

	// remember locally in block for calculation of magnetization:
	blockSpins[threadIdx.x] = spinstate;

	// store new seed values in DRAM:
	seeds[4*id    ] = z1;	// Tausworthe seeds
	seeds[4*id + 1] = z2;
	seeds[4*id + 2] = z3;
	seeds[4*id + 3] = z;	// LCRNG seed


	__syncthreads();


	// sum up this block's energy delta and magnetization:
	if(threadIdx.x == 0)
	{
		int blockEnergyDiff = 0;
		int M = 0;
		

		for(unsigned i=0; i<blockDim.x; i++)
		{
			blockEnergyDiff += deltaE[i];

			if(blockSpins[i] == -1){
				M -= 1;
				
			}
			else{
				M += 1;
				
			}
		}

		energyDifferences[blockIdx.x] += blockEnergyDiff;
		magnetization[blockIdx.x] = M;
		
	}
}



/******************************
 *  HOST FUNCTION (CPU PART)  *
 ******************************/

int main(int argc, char const **argv){
	int args = 1;

	char log_file[30];
	char hist[30];
	

	// variable responsible for storing concentration
    float x = (argc > args)?(atof(argv[args])):(0.23);
    args++;

    
    int T_cut = (argc > args)?(atoi(argv[args])):(0);
    args++;


	sprintf(log_file, "x=%0.2fT_cut=%0.2f.log", x, T_cut / 100.0);
	sprintf(hist, "Histeresis_x=%0.2f_T=%0.2f", x, T_cut / 100.0);
    std::ofstream lg(log_file);
	std::ofstream dt(hist);

	lg << "Author: William Carreras Oropesa" << std::endl;
	lg << "Simulation data" << std::endl;
	lg << "L = " << L << std::endl;
	lg << "D = " << D << std::endl;
	lg << "N = " << N << std::endl;
	lg << "J_IN = " << J_IN / 100.0 << std::endl;
	lg << "J_OU = " << J_OU / 100.0 << std::endl;
	lg << "J_PB = " << J_PB / 100.0 << std::endl;
	lg << "N_SWEEPS = " << N_SWEEPS << std::endl;

    srand48(time(NULL));

	// each spin has value 0,..,Q-1
	int spins[N];
	bool contrt[N];
    
	// calculate lattice volume elements:
	unsigned volume[D];

	for(unsigned i=0; i<=D; i++){
		
		if(i == 0)
			volume[i] = 1;
		else
			volume[i] = volume[i-1] * L;
	}	


	/* Determine the "checkerboard color" (black or white) for each site and
	   initialise lattice with random spin states: */
	unsigned w=0, b=0;
	unsigned white[N/2], black[N/2];	// store ids of white/black sites

	for(unsigned i=0; i<N; i++){
		
		// Sum of all coordinates even or odd? -> gives checkerboard color
		int csum = 0;
		for(int k=D-1; k>=0; k--)
			csum += ceil((i+1.0)/volume[k]) - 1;

		if((csum%2) == 0){	// white
		
			white[w] = i;
			w++;
		}
		else{				// black

			black[b] = i;
			b++;
		}

		// random spin state:
		spins[i] = 2 * floor(2 * drand48()) - 1;
        contrt[i] = false;
	}


	int n_ele = int(N * x);
    int n = 0;
    
    while(n < n_ele){
        
        unsigned index = static_cast<unsigned>(floor(N * drand48()));
        
        if(!contrt[index]){
           	
           	contrt[index] = true;
           	n++;
		}            	
    }


    bool* devPtrContrt;
    cudaMalloc((void**)&devPtrContrt, sizeof(contrt));
    cudaMemcpy(devPtrContrt, &contrt, sizeof(contrt), cudaMemcpyHostToDevice);
    

	// neighborhood table:
	unsigned neighbors[2*D*N];

	// calculate neighborhood table:
	for(unsigned i=0; i<N; i++){
	
		int c=0;

		for(int dim=0; dim<D; dim++){	// dimension loop
		
			for(short dir=-1; dir<=1; dir+=2){		// two directions in each dimension
			
				// neighbor's id in spin list:
				int npos = i + dir * volume[dim];

				// periodic boundary conditions:
				int test = (i % volume[dim + 1]) + dir * volume[dim];

				if(test < 0)
					npos += volume[dim + 1];
				else if(test >= volume[dim + 1])
					npos -= volume[dim + 1];
				
				neighbors[2 * D * i + c] = npos;
				c++;
			}
		}
	}


	// create 4 seed values for each thread:
	unsigned seeds[4 * N / 2];

	for(unsigned i=0; i<4*N/2; i++)
		seeds[i] = static_cast<unsigned>(4294967295 * drand48());

	// copy seeds to GPU:
	int *devPtrSeeds;
	cudaMalloc((void**)&devPtrSeeds, sizeof(seeds));
	cudaMemcpy(devPtrSeeds, &seeds, sizeof(seeds), cudaMemcpyHostToDevice);

	// copy spins to GPU:
	int *devPtrSpins;
	cudaMalloc((void**)&devPtrSpins, sizeof(spins));
	cudaMemcpy(devPtrSpins, &spins, sizeof(spins), cudaMemcpyHostToDevice);

	// copy neighborhood table to GPU:
	unsigned *devPtrNeighbors;
	cudaMalloc((void**)&devPtrNeighbors, sizeof(neighbors));
	cudaMemcpy(devPtrNeighbors, &neighbors, sizeof(neighbors), cudaMemcpyHostToDevice);

	// copy white ids to GPU:
	unsigned *devPtrWhite;
	cudaMalloc((void**)&devPtrWhite, sizeof(white));
	cudaMemcpy(devPtrWhite, &white, sizeof(white), cudaMemcpyHostToDevice);

	// copy black ids to GPU:
	unsigned *devPtrBlack;
	cudaMalloc((void**)&devPtrBlack, sizeof(black));
	cudaMemcpy(devPtrBlack, &black, sizeof(black), cudaMemcpyHostToDevice);

	// each block calculates energy difference to initial state:
	int energyDifferences[N_BLOCKS];
	for(unsigned i=0; i<N_BLOCKS; i++)
		energyDifferences[i] = 0;

	int *devPtrEnergyDifferences;
	cudaMalloc((void**)&devPtrEnergyDifferences, sizeof(energyDifferences));
	cudaMemcpy(devPtrEnergyDifferences, &energyDifferences, sizeof(energyDifferences), cudaMemcpyHostToDevice);

	// each block calculates block's magnetization:
	int magnetization_white[N_BLOCKS];
	int *devPtrMagnetization_white;
	cudaMalloc((void**)&devPtrMagnetization_white, sizeof(magnetization_white));

	int magnetization_black[N_BLOCKS];
	int *devPtrMagnetization_black;
	cudaMalloc((void**)&devPtrMagnetization_black, sizeof(magnetization_black));

	

	for(float T = 500.00; T >= T_cut; T -= 10.00){

		for(unsigned i=0; i<N_SWEEPS; i++){
			
			// White spins:
			runMetropolis<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(devPtrSeeds, devPtrSpins, devPtrNeighbors, devPtrWhite, 
								devPtrEnergyDifferences, devPtrMagnetization_white, devPtrContrt,1.0f/T, 0);

			// Black spins:
			runMetropolis<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(devPtrSeeds, devPtrSpins, devPtrNeighbors, devPtrBlack, 
								devPtrEnergyDifferences, devPtrMagnetization_black, devPtrContrt ,1.0f/T, 0);
		}
	}

	cudaMemcpy(&spins, devPtrSpins, sizeof(spins), cudaMemcpyDeviceToHost);


	// calculate energy (Potts model)
	int E = 0;

	for(unsigned i=0; i<N; i++){	
		
		for(unsigned j=0; j<2*D; j++){
		
			if(j != 4 && j !=5)
				E -= J_IN * spins[i] * spins[neighbors[2*D*i + j]];
			
			else{
				
				if(j == 4 && contrt[i])
					E -= J_PB * spins[i] * spins[neighbors[2*D*i + j]];
				else if (j == 5 && contrt[neighbors[2*D*i + j]])
					E -= J_PB * spins[i] * spins[neighbors[2*D*i + j]];
				else
					E -= J_OU * spins[i] * spins[neighbors[2*D*i + j]];
			}
		}
	}

	E /= 2; // count each interaction only once



	int E_before_simulation = E;
	long long M;	// magnetization


	for(int H = 0; H < H_max ; H++){
		
		double sum_e 		= 0;
		double sum_ee		= 0;
		double sum_m 		= 0;
		double sum_mm 		= 0;
		

		for(unsigned i=0; i<N_SWEEPS; i++){
			
			// White spins:
			runMetropolis<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(devPtrSeeds, devPtrSpins, devPtrNeighbors, devPtrWhite, 
								devPtrEnergyDifferences, devPtrMagnetization_white, devPtrContrt,1.0f/T_cut, H);

			// Black spins:
			runMetropolis<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(devPtrSeeds, devPtrSpins, devPtrNeighbors, devPtrBlack, 
								devPtrEnergyDifferences, devPtrMagnetization_black, devPtrContrt ,1.0f/T_cut, H);
			
			if(i >= 0.2*N_SWEEPS){
					
				// get energy changes from the GPU:
				cudaMemcpy(&energyDifferences, devPtrEnergyDifferences, sizeof(energyDifferences), cudaMemcpyDeviceToHost);

				// get magnetization from the GPU:
				cudaMemcpy(&magnetization_white, devPtrMagnetization_white, sizeof(magnetization_white), cudaMemcpyDeviceToHost);

				cudaMemcpy(&magnetization_black, devPtrMagnetization_black, sizeof(magnetization_black), cudaMemcpyDeviceToHost);
				
					

					E = E_before_simulation;
					M = 0;

					for(unsigned t=0; t<N_BLOCKS; t++){	// take energy changes into account
				
						E += energyDifferences[t];
						M += magnetization_white[t] + magnetization_black[t];
					}

					double m = static_cast<double>(M) / static_cast<double>(N);
					double e = static_cast<double>(E) / static_cast<double>(N);

					sum_e 	 += e;
					sum_ee 	 += e*e;
				
					sum_m 	 += m;
					sum_mm   += m*m;
			}
		}

		double beta = 1.0f / T_cut;

		double mE_Ising  = sum_e / (0.8*N_SWEEPS);
		double mEE_Ising = sum_ee / (0.8*N_SWEEPS);
		double C_Ising 	 = beta*beta*(mEE_Ising - mE_Ising*mE_Ising);

		C_Ising *= static_cast<double>(N);
		
		
		double m_M    = static_cast<double>(sum_m) / (0.8*N_SWEEPS);
		double m_MM   = static_cast<double>(sum_mm) / (0.8*N_SWEEPS);


		// magnetization:
		double chi_uniform = beta*(m_MM - m_M*m_M);
		

		dt << H / 100.0  << "\t" << m_M << "\t" << chi_uniform << std::endl;
	}

	for(int H = H_max; H > H_min ; H--){
		
		double sum_e 		= 0;
		double sum_ee		= 0;
		double sum_m 		= 0;
		double sum_mm 		= 0;
		

		for(unsigned i=0; i<N_SWEEPS; i++){
			
			// White spins:
			runMetropolis<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(devPtrSeeds, devPtrSpins, devPtrNeighbors, devPtrWhite, 
								devPtrEnergyDifferences, devPtrMagnetization_white, devPtrContrt,1.0f/T_cut, H);

			// Black spins:
			runMetropolis<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(devPtrSeeds, devPtrSpins, devPtrNeighbors, devPtrBlack, 
								devPtrEnergyDifferences, devPtrMagnetization_black, devPtrContrt ,1.0f/T_cut, H);
			
			if(i >= 0.2*N_SWEEPS){
					
				// get energy changes from the GPU:
				cudaMemcpy(&energyDifferences, devPtrEnergyDifferences, sizeof(energyDifferences), cudaMemcpyDeviceToHost);

				// get magnetization from the GPU:
				cudaMemcpy(&magnetization_white, devPtrMagnetization_white, sizeof(magnetization_white), cudaMemcpyDeviceToHost);

				cudaMemcpy(&magnetization_black, devPtrMagnetization_black, sizeof(magnetization_black), cudaMemcpyDeviceToHost);
				
					

					E = E_before_simulation;
					M = 0;

					for(unsigned t=0; t<N_BLOCKS; t++){	// take energy changes into account
				
						E += energyDifferences[t];
						M += magnetization_white[t] + magnetization_black[t];
					}

					double m = static_cast<double>(M) / static_cast<double>(N);
					double e = static_cast<double>(E) / static_cast<double>(N);

					sum_e 	 += e;
					sum_ee 	 += e*e;
				
					sum_m 	 += m;
					sum_mm   += m*m;
				}
			}

			double beta = 1.0f / T_cut;

			double mE_Ising  = sum_e / (0.8*N_SWEEPS);
			double mEE_Ising = sum_ee / (0.8*N_SWEEPS);
			double C_Ising 	 = beta*beta*(mEE_Ising - mE_Ising*mE_Ising);

			C_Ising *= static_cast<double>(N);
		
		
			double m_M    = static_cast<double>(sum_m) / (0.8*N_SWEEPS);
			double m_MM   = static_cast<double>(sum_mm) / (0.8*N_SWEEPS);


			// magnetization:
			double chi_uniform = beta*(m_MM - m_M*m_M);
		

			dt << H / 100.0  << "\t"  << m_M << "\t" << chi_uniform << std::endl;
	}

	for(int H = H_min; H <= H_max ; H++){
		
		double sum_e 		= 0;
		double sum_ee		= 0;
		double sum_m 		= 0;
		double sum_mm 		= 0;
		

		for(unsigned i=0; i<N_SWEEPS; i++){
			
			// White spins:
			runMetropolis<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(devPtrSeeds, devPtrSpins, devPtrNeighbors, devPtrWhite, 
								devPtrEnergyDifferences, devPtrMagnetization_white, devPtrContrt,1.0f/T_cut, H);

			// Black spins:
			runMetropolis<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(devPtrSeeds, devPtrSpins, devPtrNeighbors, devPtrBlack, 
								devPtrEnergyDifferences, devPtrMagnetization_black, devPtrContrt ,1.0f/T_cut, H);
			
			if(i >= 0.2*N_SWEEPS){
					
				// get energy changes from the GPU:
				cudaMemcpy(&energyDifferences, devPtrEnergyDifferences, sizeof(energyDifferences), cudaMemcpyDeviceToHost);

				// get magnetization from the GPU:
				cudaMemcpy(&magnetization_white, devPtrMagnetization_white, sizeof(magnetization_white), cudaMemcpyDeviceToHost);

				cudaMemcpy(&magnetization_black, devPtrMagnetization_black, sizeof(magnetization_black), cudaMemcpyDeviceToHost);
				
					

					E = E_before_simulation;
					M = 0;

					for(unsigned t=0; t<N_BLOCKS; t++){	// take energy changes into account
				
						E += energyDifferences[t];
						M += magnetization_white[t] + magnetization_black[t];
					}

					double m = static_cast<double>(M) / static_cast<double>(N);
					double e = static_cast<double>(E) / static_cast<double>(N);

					sum_e 	 += e;
					sum_ee 	 += e*e;
				
					sum_m 	 += m;
					sum_mm   += m*m;
				}
			}

			double beta = 1.0f / T_cut;

			double mE_Ising  = sum_e / (0.8*N_SWEEPS);
			double mEE_Ising = sum_ee / (0.8*N_SWEEPS);
			double C_Ising 	 = beta*beta*(mEE_Ising - mE_Ising*mE_Ising);

			C_Ising *= static_cast<double>(N);
		
		
			double m_M    = static_cast<double>(sum_m) / (0.8*N_SWEEPS);
			double m_MM   = static_cast<double>(sum_mm) / (0.8*N_SWEEPS);


			// magnetization:
			double chi_uniform = beta*(m_MM - m_M*m_M);
		

			dt << H / 100.0  << "\t" << m_M << "\t" << chi_uniform << std::endl;
	}

	

	cudaFree(devPtrSeeds);
	cudaFree(devPtrSpins);
	cudaFree(devPtrNeighbors);
	cudaFree(devPtrWhite);
	cudaFree(devPtrBlack);
	cudaFree(devPtrEnergyDifferences);
	cudaFree(devPtrContrt);
	cudaFree(devPtrMagnetization_white);
	cudaFree(devPtrMagnetization_black);

	lg.close();
	dt.close();

    

	return 0;
}
