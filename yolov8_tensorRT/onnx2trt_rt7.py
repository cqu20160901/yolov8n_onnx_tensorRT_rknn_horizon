import tensorrt as trt

G_LOGGER = trt.Logger()

batch_size = 1
imput_h = 640
imput_w = 640


def get_engine(onnx_model_name, trt_model_name):
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network,
                                                                                                             G_LOGGER) as parser:
        builder.max_batch_size = batch_size
        builder.max_workspace_size = 2 << 30
        print('Loading ONNX file from path {}...'.format(onnx_model_name))
        with open(onnx_model_name, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_model_name))

        ####
        # builder.int8_mode = True
        # builder.int8_calibrator = calib
        builder.fp16_mode = True
        ####

        print("num layers:", network.num_layers)
        # last_layer = network.get_layer(network.num_layers - 1)
        # if not last_layer.get_output(0):
        # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))//有的模型需要，有的模型在转onnx的之后已经指定了，就不需要这行

        network.get_input(0).shape = [batch_size, 3, imput_h, imput_w]
        engine = builder.build_cuda_engine(network)
        print("engine:", engine)
        print("Completed creating Engine")
        with open(trt_model_name, "wb") as f:
            f.write(engine.serialize())
        return engine


def main():
    onnx_file_path = './yolov8n_ZQ.onnx'
    engine_file_path = './yolov8n_ZQ.trt'

    engine = get_engine(onnx_file_path, engine_file_path)


if __name__ == '__main__':
    print("This is main ...")
    main()
