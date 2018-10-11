/* THRUST - Example REDUCE - by default it computes the sum. In host or device
 __host__ __device__ thrust::reduce() {...} */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>						// rand
#include <thrust/reduce.h>

#define N 512

int main(void)
{
	// Generate random data on the host
	thrust::host_vector<int> h_vec(N);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	// Copy data to device
	thrust::device_vector<int> d_vec = h_vec;

	// Compute sum on HOST
	int h_sum = thrust::reduce(h_vec.begin(), h_vec.end());

	// Compute sum on DEVICE
	int d_sum = thrust::reduce(d_vec.begin(), d_vec.end());

	// Print the sum
	std::cout << h_sum << " = " << d_sum << std::endl;
	
	return 0;
}