#include "miopen.hpp"

#include "tensor.hpp"

#include "utils.hpp"

#include "layers.hpp"

#include "multi_layers.hpp"




void MobileNet() {

    TensorDesc input_dim(64, 3, 224, 224);
	
    Sequential features(input_dim);


    /* features */

	features.addConv(32, 3, 1, 2);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(32, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(64, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(64, 3, 1, 2);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(128, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(128, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(128, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(128, 3, 1, 2);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(256, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(256, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(256, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(256, 3, 1, 2);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(512, 3, 1, 2);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(1024, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(1024, 3, 1, 1);
	features.addBatchNorm();
	features.addReLU();

	features.addConv(1024, 1, 0, 1);
	features.addBatchNorm();
	features.addReLU();


	features.addMaxPool(1, 0, 1);

	features.addConv(1000, 1, 0, 1);

 
    Sequential classifier(features.getOutputDesc());

    classifier.reshape(input_dim.n, 1000 * 7 * 7, 1, 1);
	
	Model m(input_dim);


    m.add(features);

    m.add(classifier);



    BenchmarkLogger::new_session("mobile_net");

    BenchmarkLogger::benchmark(m, 1000);

}





int main(int argc, char *argv[])

{

    device_init();

    // enable profiling

    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    MobileNet();

    miopenDestroy(mio::handle());

    return 0;

}


