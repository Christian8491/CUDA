/* THRUST - Example for_each
 Compute the negative of each element in the container (inplace) 
 for_each is equivalent to: 
 while ( h_vec.begin() != h_vec.end() ) {...} */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/for_each.h>

#define N 1028

template<typename T>
struct negative
{
	void operator()(T& x) const { x = -x; }
};

int main(void)
{
	// Generate random data (integers) on the host
	thrust::host_vector<int> h_vec(N);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	// Last parameter is a functor
	thrust::for_each(h_vec.begin(), h_vec.end(), negative<int>());

	return 0;
}