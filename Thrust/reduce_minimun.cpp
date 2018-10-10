/* CUDA - Example 2 - REDUCE with THRUST library
thrust::reduce(): need some params and a functor, for this case te minimun functor
negate function that change the sign of all elements
extension .cu */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>				// rand
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cstdlib>
#include <ctime>

__host__ int main()
{
	/* Create two vectors in host and fill in the first */
	thrust::host_vector<float> h_vec(6);
	thrust::host_vector<float> h_aux(h_vec.size());
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	/* Create two vectors in device */
	thrust::device_vector<float> d_vec = h_vec;
	thrust::device_vector<float> d_temp(d_vec.size());
	
	/* transform need a functor, in this case the negate function */
	thrust::transform(d_vec.begin(), d_vec.end(), d_temp.begin(), thrust::negate<float>());

	/* return the minimun value of d_temp */
	float minValue = thrust::reduce(d_temp.begin(), d_temp.end(), (float)d_temp[0], thrust::minimum<float>());

	printf("min value: %.4f\n", minValue);

	return 0;
}