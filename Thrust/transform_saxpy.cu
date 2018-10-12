/* THRUST - Example saxpy using thrust::transform()
 Saxpy function: z = a * x + y 
 thrust::transform() can works in host or device
 thrust::sequence() can works in host or device  */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/functional.h>						// negate

#define N 1024

template<typename T>
struct saxpy_functor
{
	const T a;

	saxpy_functor(T _a) : a(_a){}

	__host__ __device__ 
	T operator()(const T& x, const T& y) const { return a * x + y; }
};

int main(void)
{
	//--------- HOST Example ----------
	thrust::host_vector<int> x_host(N);
	thrust::host_vector<int> y_host(N);
	thrust::host_vector<int> z_host(N);
	
	// Initialize x_host: [0, 1, 2, .., N-1]
	thrust::sequence(x_host.begin(), x_host.end());

	// Compute y_host = -x_host
	thrust::transform(x_host.begin(), x_host.end(), y_host.begin(), thrust::negate<int>());

	// Compute saxpy in host
	saxpy_functor<int> functor_host(4);
	thrust::transform(x_host.begin(), x_host.end(), y_host.begin(), z_host.begin(), functor_host);

	// To see if it works
	std::cout << z_host[2] << std::endl;


	//--------- DEVICE Example ----------
	thrust::device_vector<int> x_dev = x_host;
	thrust::device_vector<int> y_dev = y_host;
	thrust::device_vector<int> z_dev = z_host;

	// Compute saxpy in device
	saxpy_functor<int> functor_dev(6);
	thrust::transform(x_dev.begin(), x_dev.end(), y_dev.begin(), z_dev.begin(), functor_dev);

	// Copy to a new Z_host container
	thrust::host_vector<int> Z_host(N);
	thrust::copy(z_dev.begin(), z_dev.end(), Z_host.begin());

	// To see if it works
	std::cout << Z_host[2] << std::endl;

	return 0;
}