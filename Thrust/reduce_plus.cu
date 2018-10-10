/* THRUST - Example REDUCE PLUS for a single device_vector
thrust::reduce(): need some params and a function object (functor) in the last */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>						// rand
#include <thrust/reduce.h>

#define N 128

int main(void)
{
	// Generate random data on the host
	thrust::host_vector<int> h_vec(N);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	// Transfer data to device
	thrust::device_vector<int> d_vec = h_vec;

	// The third parameter is the initial value
	int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

	// Print the sum
	std::cout << "Exit sum: " << sum << std::endl;

	return 0;
}