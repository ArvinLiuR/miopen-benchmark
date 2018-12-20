#include "miopen.hpp"

#include "tensor.hpp"

#include "utils.hpp"

#include "layers.hpp"

#include "multi_layers.hpp"




void VGG19() {

    TensorDesc input_dim(32, 3, 224, 224);

    Sequential features(input_dim);

    /* features */
	features.addConv(64, 3, 1, 1);
	features.addReLU();

	features.addConv(64, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);


	features.addConv(128, 3, 1, 1);
	features.addReLU();

	features.addConv(128, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);


	features.addConv(256, 3, 1, 1);
	features.addReLU();

	features.addConv(256, 3, 1, 1);
	features.addReLU();

	features.addConv(256, 3, 1, 1);
	features.addReLU();

    features.addConv(256, 3, 1, 1);
    features.addReLU();

    features.addMaxPool(2, 0, 2);

	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();

    features.addConv(512, 3, 1, 1);
    features.addReLU();

	features.addMaxPool(2, 0, 2);

	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();

    features.addConv(512, 3, 1, 1);
    features.addReLU();

	features.addMaxPool(2, 0, 2);


	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
    features.addConv(512, 3, 1, 1);
    features.addReLU();


    /* classifier */

    Sequential classifier(features.getOutputDesc());

    // TODO Dropout

    classifier.reshape(input_dim.n, 512 * 7 * 7, 1, 1);

    classifier.addLinear(4096);

    classifier.addReLU();

    // TODO: Dropout

    classifier.addLinear(4096);

    classifier.addReLU();

    classifier.addLinear(1000);



    Model m(input_dim);

    m.add(features);

    m.add(classifier);



    BenchmarkLogger::new_session("VGG_19");

    BenchmarkLogger::benchmark(m, 50);

}




int main(int argc, char *argv[])

{

    device_init();

    // enable profiling

    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    VGG19();

    miopenDestroy(mio::handle());

    return 0;

}
