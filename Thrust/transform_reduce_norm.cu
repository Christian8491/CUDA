/* THRUST - Example TRANSFORM_REDUCE - Compute the norm of a vector
thrust::transform_reduce (InputIterator first, InputIterator last, UnaryFunction unary_op, 
							OutputType init, BinaryFunction binary_op)
The binary operation has to be associative and commutative! 
thrust::transform_reduce() worsk in both host and device */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <cmath>										// sqrt

#define N 4

template<typename T>
struct square_function
{
	__host__ __device__
	T operator()(const T& x) const { return x * x; }
};

int main(void)
{
	// Generate a sequence of N values on the host [0, 1, 2, .., N-1]
	thrust::host_vector<int> h_vec(N);
	thrust::sequence(h_vec.begin(), h_vec.end());

	// Generate a sequence of N values on the device [1, 2, 3, .., N]
	thrust::device_vector<int> d_vec(N);
	thrust::sequence(d_vec.begin(), d_vec.end(), 1);

	// functors (works in both host and device)
	square_function<int> functor_square;
	thrust::plus<int> functor_sum;

	// transform_reduce for host
	int norm_host = thrust::transform_reduce(h_vec.begin(), h_vec.end(), functor_square, 0, functor_sum);

	// transform_reduce for device
	int norm_device = thrust::transform_reduce(d_vec.begin(), d_vec.end(), functor_square, 0, functor_sum);

	// Print products for host and device
	std::cout << "norm host: " << sqrt(norm_host) << std::endl;
	std::cout << "norm device: " << sqrt(norm_device) << std::endl;

	return 0;
}