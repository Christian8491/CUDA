/* THRUST - Example SORT */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>				// rand
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <ctime>

__host__ int main()
{
	/* Step 1: Elapsed time to generate random data */
	clock_t begin_host = clock();
	thrust::host_vector<int> h_vec(5000000);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);
	clock_t end_host = clock();
	double elapsedTime = float(end_host - begin_host) / CLOCKS_PER_SEC;
	printf("\nTime to generate data in CPU: %.5f\n", elapsedTime);


	/* Step 2: Elapsed time for copy data from CPU to GPU */
	clock_t begin_device = clock();
	thrust::device_vector<int> d_vec = h_vec;
	clock_t end_device = clock();
	elapsedTime = float(end_device - begin_device) / CLOCKS_PER_SEC;
	printf("\nTime to pass data from CPU to GPU: %.5f\n", elapsedTime);


	/* Step 3: Elapsed time to sort on the GPU and copy back */
	clock_t begin = clock();
	thrust::sort(d_vec.begin(), d_vec.end());
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
	clock_t end = clock();
	elapsedTime = float(end - begin) / CLOCKS_PER_SEC;
	printf("\nTotal time to sort: %.5f\n", elapsedTime);

	return 0;
}