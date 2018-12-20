#include "miopen.hpp"

#include "tensor.hpp"

#include "utils.hpp"

#include "layers.hpp"

#include "multi_layers.hpp"



void GoogleNet() {

    TensorDesc input_dim(64, 3, 224, 224);
    Sequential features(input_dim);
	
    /* features */
	features.addConv(64, 7, 3, 2);
	features.addReLU();
	features.addMaxPool(3, 0, 2);

	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(192, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(3, 0, 2);

	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(128, 3, 1, 1);
	features.addReLU();
	features.addConv(16, 1, 1, 1);
	features.addReLU();
	features.addConv(32, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(32, 1, 1, 1);
	features.addReLU();
	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addConv(192, 3, 1, 1);
	features.addReLU();
	features.addConv(32, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 0, 2);

	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(208, 3, 1, 1);
	features.addReLU();
	features.addConv(16, 1, 1, 1);
	features.addReLU();
	features.addConv(48, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

    features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(5, 0, 3);


	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(7, 0, 1);

	features.addConv(1024, 1, 1, 1);
	features.addReLU();
	features.addConv(1000, 1, 1, 1);
	features.addReLU();


	features.addConv(160, 1, 1, 1);
	features.addReLU();
	features.addConv(112, 1, 1, 1);
	features.addReLU();
	features.addConv(224, 3, 1, 1);
	features.addReLU();
	features.addConv(24, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 3, 1, 1);
	features.addReLU();
	features.addConv(24, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(112, 1, 1, 1);
	features.addReLU();
	features.addConv(144, 1, 1, 1);
	features.addReLU();
	features.addConv(288, 3, 1, 1);
	features.addReLU();
	features.addConv(32, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(5, 0, 3);

	features.addConv(128, 1, 1, 1);
	features.addReLU();
	
    features.addConv(1024, 1, 1, 1);
	features.addReLU();
	features.addConv(1000, 1, 1, 1);
	features.addReLU();


	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(160, 1, 1, 1);
	features.addReLU();
	features.addConv(320, 3, 1, 1);
	features.addReLU();
	features.addConv(32, 1, 1, 1);
	features.addReLU();
	features.addConv(128, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 0, 2);

	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(160, 1, 1, 1);
	features.addReLU();
	features.addConv(320, 3, 1, 1);
	features.addReLU();
	features.addConv(32, 1, 1, 1);
	features.addReLU();
	features.addConv(128, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addConv(384, 1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
    features.addConv(384, 3, 1, 1);
	features.addReLU();
	features.addConv(48, 1, 1, 1);
	features.addReLU();
	features.addConv(128, 5, 2, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(7, 0, 1);

	
	features.addConv(1000, 1, 1, 1);

    Model m(input_dim);


    m.add(features);


    BenchmarkLogger::new_session("Google_Net");

    BenchmarkLogger::benchmark(m, 1000);

}




int main(int argc, char *argv[])

{

    device_init();

    // enable profiling

    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    GoogleNet();

    miopenDestroy(mio::handle());

    return 0;

}
