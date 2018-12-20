#include "miopen.hpp"

#include "tensor.hpp"

#include "utils.hpp"

#include "layers.hpp"

#include "multi_layers.hpp"




void Inception_v4() {

    TensorDesc input_dim(4, 4, 299, 299);

    Sequential features(input_dim);
	
    /* classifier */
    Sequential classifier(features.getOutputDesc());

    /* features */
	features.addConv(32, 3, 1, 2);
	features.addReLU();
	features.addConv(32, 3, 1, 1);
	features.addReLU();
	features.addConv(64, 3, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 2);
	features.addReLU();
    features.addMaxPool(3, 0, 2);

	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(192, 3, 0, 2);
	features.addReLU();
	features.addMaxPool(3, 0, 2);

	features.addMaxPool(3, 1, 1);

	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 1, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(64, 1, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addConv(96, 3, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 0, 2);

	features.addConv(384, 3, 1, 2);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224, 3, 1, 1);
	features.addReLU();
	features.addConv(256, 3, 1, 2);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addConv(384, 1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224, 1, 1, 1);
	features.addReLU();
	features.addConv(224, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128,  1, 1, 1);
	features.addReLU();
	features.addConv(384,  1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(224, 1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addConv(384, 1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128,  1, 1, 1);
	features.addReLU();
	features.addConv(384,  1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128,  1, 1, 1);
	features.addReLU();
	features.addConv(384,  1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128, 1, 1, 1);
	features.addReLU();
	features.addConv(384,  1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(224, 1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(128,  1, 1, 1);
	features.addReLU();
	features.addConv(384,  1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(192,  1, 1, 1);
	features.addReLU();
	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(224,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 0, 2);

	features.addConv(192, 1, 1, 1);
	features.addReLU();
	features.addConv(192, 3, 1, 2);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(320, 1, 1, 1);
	features.addReLU();
	features.addConv(320, 3, 1, 2);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(384,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(384, 1, 1, 1);
	features.addReLU();
	features.addConv(448, 1, 1, 1);
	features.addReLU();
	features.addConv(512,  1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(384,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(384, 1, 1, 1);
	features.addReLU();
	features.addConv(448, 1, 1, 1);
	features.addReLU();
	features.addConv(512,  1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(3, 1, 1);

	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(384,  1, 1, 1);
	features.addReLU();
	features.addConv(256,  1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(384, 1, 1, 1);
	features.addReLU();
	features.addConv(448, 1, 1, 1);
	features.addReLU();
	features.addConv(512,  1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addConv(256, 1, 1, 1);
	features.addReLU();
	features.addMaxPool(1, 0, 1);


	Model m(input_dim);
	
    m.add(features);
    m.emplace<Reshape>(input_dim.n, m.last_output_dim().c * m.last_output_dim().h * m.last_output_dim().w, 1, 1);
	m.emplace<Linear>(101);
 

    BenchmarkLogger::new_session("Inception_v4");

    BenchmarkLogger::benchmark(m, 1000);

}




int main(int argc, char *argv[])

{

    device_init();

    // enable profiling

    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    Inception_v4();

    miopenDestroy(mio::handle());

    return 0;

}