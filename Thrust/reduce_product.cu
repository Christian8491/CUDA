/* THRUST - Example REDUCE PRODUCT - Calculate the product of a sequence.
thrust::reduce() worsk in both host and device */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>						// rand
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#define N 4

int main(void)
{
	// Generate random data on the host
	thrust::host_vector<int> h_vec(N);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	// Generate a sequence of N values since 1 [1, 2, .., N ]
	thrust::device_vector<int> d_vec(N);
	thrust::sequence(d_vec.begin(), d_vec.end(), 1);

	// reduce for host
	int product_host = thrust::reduce(h_vec.begin(), h_vec.end(), 1, thrust::multiplies<int>());

	// reduce for device
	int product_device = thrust::reduce(d_vec.begin(), d_vec.end(), 1, thrust::multiplies<int>());

	// Print products for host and device
	std::cout << "product host: " << product_host << std::endl;
	std::cout << "product host: " << product_device << std::endl;

	return 0;
}