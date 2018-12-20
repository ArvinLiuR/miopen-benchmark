#include "miopen.hpp"

#include "tensor.hpp"

#include "utils.hpp"

#include "layers.hpp"

#include "multi_layers.hpp"






Sequential makeBlock(const TensorDesc& input_dim, int planes, int stride=1) {

    // BasicBlock

    DEBUG("making block with dim " << input_dim );

    Sequential preblock(input_dim, "Shortcut_F");

    preblock.emplace<ConvLayer>(planes, 3, 1, stride);

    preblock.emplace<BatchNorm>();

    preblock.emplace<ReLU>();

    preblock.emplace<ConvLayer>(planes, 3, 1, 1);

    preblock.emplace<BatchNorm>();



    Sequential downsample_block(input_dim, "downsample");

    downsample_block.emplace<ConvLayer>(planes, 1, 0, stride);

    downsample_block.emplace<BatchNorm>();



    Sequential block(input_dim);

    ShortCutAdd s(input_dim);

    s.setF(preblock);

    if (stride != 1) {

        s.setG(downsample_block);

    }

    block.add(s);

    block.emplace<ReLU>();

    return block;

}



Sequential makeBottleneck(const TensorDesc& input_dim, int planes, int stride=1) {

    Sequential left(input_dim, "Shortcut_F");

    left.emplace<ConvLayer>(planes, 1);

    left.emplace<BatchNorm>();

    left.emplace<ReLU>();

    // reduce size by `stride` (either 1 or 2)

    left.emplace<ConvLayer>(planes, 3, 1, stride);

    left.emplace<BatchNorm>();

    left.emplace<ReLU>();

    // 4x expansion of channels

    left.emplace<ConvLayer>(4*planes, 1);

    left.emplace<BatchNorm>();



    // downsample residual

    Sequential downsample_block(input_dim, "downsample");

    downsample_block.emplace<ConvLayer>(4*planes, 1, 0, stride);

    downsample_block.emplace<BatchNorm>();



    Sequential block(input_dim);

    ShortCutAdd s(input_dim);

    s.setF(left);

    if (stride != 1 || input_dim.c != planes*4) {

        s.setG(downsample_block);

    }

    block.add(s);

    block.emplace<ReLU>();

    return block;

}



template <typename F>

Sequential makeLayer(const TensorDesc& input_dim, int planes, int blocks, int stride, F blockFunc) {

    Sequential layer(input_dim);



    // add one downsample block inplanes -> planes, stride

    layer.add(blockFunc(layer.getOutputDesc(), planes, stride));



    for (int i = 1; i < blocks; ++i) {

        layer.add(blockFunc(layer.getOutputDesc(), planes, 1));

    }

    return layer;

}





template <typename B>

Model make_resneXt(const TensorDesc& input_dim, B blockfunc, const std::vector<int>& layers, int num_classes = 1000) {

    Model m(input_dim);



    Sequential pre(input_dim, "ResNeXt Pre");

    pre.emplace<ConvLayer>(64, 7, 3, 2);

    pre.emplace<BatchNorm>();

    pre.emplace<ReLU>();

    pre.emplace<MaxPool>(3, 0, 2);

    DEBUG("ResNeXt Pre output dims: " << pre.getOutputDesc());



    m.add(pre);

    m.add(makeLayer(m.getOutputDesc(), 128,  layers[0], 1, blockfunc));

    m.add(makeLayer(m.getOutputDesc(), 256, layers[1], 2, blockfunc));

    m.add(makeLayer(m.getOutputDesc(), 512, layers[2], 2, blockfunc));

    m.add(makeLayer(m.getOutputDesc(), 1024, layers[3], 2, blockfunc));

    m.emplace<AvgPool>(7, 0, 1);

    m.emplace<Reshape>(input_dim.n, m.last_output_dim().c * m.last_output_dim().h * m.last_output_dim().w, 1, 1);

    m.emplace<Linear>(1000);

    return m;

}



Model resneXt(const std::string& name) {

    // batch_size = 32 per gpu

    TensorDesc input_dim(16, 3, 224, 224);



    if (name == "resneXt18") {

        return make_resneXt(input_dim, &makeBlock, {2, 2, 2, 2});

    } else if (name == "resneXt34") {

        return make_resneXt(input_dim, &makeBlock, {3, 4, 6, 3});

    } else if (name == "resneXt50") {

        return make_resneXt(input_dim, &makeBottleneck, {3, 4, 6, 3});

    } else if (name == "resneXt101") {

        return make_resneXt(input_dim, &makeBottleneck, {3, 4, 23, 3});

    } else if (name == "resneXt152") {

        return make_resneXt(input_dim, &makeBottleneck, {3, 8, 36, 3});

    } else {

        return Model(input_dim);

    }

}



int main(int argc, char *argv[])

{

    device_init();

    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));



    std::string mname = "resneXt50";

    Model m = resneXt(mname);

    BenchmarkLogger::new_session(mname);

    BenchmarkLogger::benchmark(m, 50);

}