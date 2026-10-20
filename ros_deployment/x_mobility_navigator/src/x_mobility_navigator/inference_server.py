"""Python 3 TensorRT service; this process has no ROS dependencies."""

import argparse
import logging
import os
import signal
import socket
import stat

import numpy as np

from .inference_client import receive_tensors, send_tensors

INPUT_NAMES = {
    'image', 'route', 'speed', 'action_input', 'history_input', 'sample_input'
}
OUTPUT_SIZES = {
    'action_output': 6,
    'path_output': 10,
    'history_output': 1024,
    'sample_output': 512,
}


class TensorRTRunner:
    def __init__(self, engine_path):
        # Import CUDA only in the inference process, on the serving thread.
        import pycuda.autoinit  # pylint: disable=unused-import,import-outside-toplevel
        import pycuda.driver as cuda  # pylint: disable=import-outside-toplevel
        import tensorrt as trt  # pylint: disable=import-outside-toplevel

        if trt.__version__.split('.')[0] != '8':
            raise RuntimeError('This service requires TensorRT 8.x')
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError('Could not deserialize engine: ' + engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Could not create TensorRT execution context')
        self.shapes = {}
        self.device_buffers = {}
        self.outputs = {}
        self.bindings = []
        actual_inputs, actual_outputs = set(), set()
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            shape = tuple(self.engine.get_binding_shape(index))
            if any(dimension <= 0 for dimension in shape):
                raise ValueError('Export fixed input/output shapes; got ' + str(shape))
            if np.dtype(trt.nptype(self.engine.get_binding_dtype(index))) != np.float32:
                raise ValueError('Engine bindings must be float32: ' + name)
            self.shapes[name] = shape
            buffer = cuda.mem_alloc(int(np.prod(shape)) * 4)
            self.device_buffers[name] = buffer
            self.bindings.append(int(buffer))
            if self.engine.binding_is_input(index):
                actual_inputs.add(name)
            else:
                actual_outputs.add(name)
                self.outputs[name] = np.empty(shape, dtype=np.float32)
        if actual_inputs != INPUT_NAMES or actual_outputs != set(OUTPUT_SIZES):
            raise ValueError('Engine must have the default six inputs and four outputs')
        for name, size in OUTPUT_SIZES.items():
            if self.outputs[name].size != size:
                raise ValueError('Unexpected output size: ' + name)

    def infer(self, inputs):
        if set(inputs) != INPUT_NAMES:
            raise ValueError('Unexpected inference input names')
        for name in INPUT_NAMES:
            value = np.ascontiguousarray(inputs[name], dtype=np.float32)
            if value.size != int(np.prod(self.shapes[name])):
                raise ValueError('Input size does not match engine: ' + name)
            self.cuda.memcpy_htod(self.device_buffers[name], value)
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError('TensorRT inference failed')
        for name, value in self.outputs.items():
            self.cuda.memcpy_dtoh(value, self.device_buffers[name])
        return self.outputs


def serve(runner, socket_path):
    if os.path.lexists(socket_path):
        if not stat.S_ISSOCK(os.stat(socket_path).st_mode):
            raise RuntimeError('Socket path already contains a regular file')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            try:
                probe.connect(socket_path)
            except ConnectionRefusedError:
                os.unlink(socket_path)
            else:
                raise RuntimeError('An inference server is already using ' + socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(socket_path)
        os.chmod(socket_path, 0o600)
        listener.listen(1)
        logging.info('Inference ready on %s', socket_path)
        try:
            while True:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(30.0)
                    try:
                        while True:
                            inputs = receive_tensors(connection)
                            send_tensors(connection, runner.infer(inputs))
                    except IOError as ex:
                        logging.info('Client disconnected: %s', ex)
                    except (ValueError, KeyError, TypeError, RuntimeError):
                        logging.exception('Invalid request or inference failed')
        finally:
            os.unlink(socket_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--socket', default='/tmp/x-mobility/inference.sock')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        serve(TensorRTRunner(args.engine), args.socket)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
