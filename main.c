#include <stdio.h>
#include <omp.h>

int main(void) {
	omp_set_num_threads(4);
	
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int np = omp_get_num_threads();
		printf("Hello world %d of %d\n", id, np);
	}
	return 0;
};