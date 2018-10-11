/* To see the update version of THRUST */

#include <thrust/version.h>
#include <iostream>

int main(void)
{
	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust version: " << major << "." << minor << std::endl;

	return 0;
}