/* CUDA - Example 2 - TRANSFORM with THRUST library
thrust::transform(): need a object function (functor), for this case we use the
negate function that change the sign of all elements
extension .cu */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>				// rand
#include <thrust/copy.h>
#include <cstdlib>
#include <ctime>

__host__ int main()
{
	/* Create two vectors in host and fill in the first */
	thrust::host_vector<float> h_vec(50000);
	thrust::host_vector<float> h_aux(h_vec.size());
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	/* Create two vectors in device */
	thrust::device_vector<float> d_vec = h_vec;
	thrust::device_vector<float> d_temp(d_vec.size());
	
	/* transform need a functor, in this case the negate function */
	thrust::transform(d_vec.begin(), d_vec.end(), d_temp.begin(), thrust::negate<float>());
	
	thrust::copy(d_temp.begin(), d_temp.end(), h_aux.begin());

	printf("Host 0 element: %.4f\n", h_vec[0]);
	printf("Host aux 0 element: %.4f\n", h_aux[0]);

	system("pause");

	return 0;
}