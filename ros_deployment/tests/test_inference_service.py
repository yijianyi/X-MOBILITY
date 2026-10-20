"""Real Unix socket transport tests and TensorRT contract tests without a GPU."""

import importlib
import json
import multiprocessing
import os
import socket
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'x_mobility_navigator/src'
sys.path.insert(0, str(SOURCE_ROOT))
transport = importlib.import_module('x_mobility_navigator.inference_client')
server = importlib.import_module('x_mobility_navigator.inference_server')


class EchoRunner:
    def infer(self, inputs):
        return {name + '_reply': array + 1 for name, array in inputs.items()}


class MemoryConnection:
    """Exercise the wire format even where socket I/O is sandboxed."""

    def __init__(self, data=b''):
        self.data = bytearray(data)
        self.offset = 0

    def sendall(self, data):
        self.data.extend(data)

    def recv_into(self, view, length):
        count = min(length, 7, len(self.data) - self.offset)
        view[:count] = self.data[self.offset:self.offset + count]
        self.offset += count
        return count


class TransportTests(unittest.TestCase):
    def test_float32_arrays_round_trip_over_fragmented_stream(self):
        inputs = {'image': np.arange(3 * 256 * 256, dtype=np.float32).reshape(3, 256, 256),
                  'speed': np.array(0.75, dtype=np.float32),
                  'route': np.arange(76, dtype=np.float32).reshape(19, 4)}
        connection = MemoryConnection()
        transport.send_tensors(connection, inputs)
        outputs = transport.receive_tensors(connection)
        for name in inputs:
            np.testing.assert_array_equal(outputs[name].ravel(), inputs[name].ravel())
            self.assertEqual(outputs[name].dtype, np.float32)

    def test_closed_socket_is_reported(self):
        with self.assertRaises(IOError):
            transport.receive_tensors(MemoryConnection())

    def test_invalid_metadata_and_payload_sizes_are_rejected(self):
        cases = [
            [],
            [{'name': 'a', 'shape': [2], 'nbytes': 4}],
            [{'name': 'a', 'shape': [-1], 'nbytes': -4}],
            [{'name': 'a', 'shape': [True], 'nbytes': 4}],
            [{'name': 'a', 'shape': [transport.MAX_PAYLOAD_BYTES],
              'nbytes': transport.MAX_PAYLOAD_BYTES * 4}],
            [{'name': 'a', 'shape': [1], 'nbytes': 4}] * 2,
        ]
        for metadata in cases:
            with self.subTest(metadata=metadata):
                encoded = json.dumps(metadata).encode('utf-8')
                connection = MemoryConnection(struct.pack('!I', len(encoded)) + encoded)
                with self.assertRaises(ValueError):
                    transport.receive_tensors(connection)

    def test_oversized_header_is_rejected_before_allocation(self):
        connection = MemoryConnection(struct.pack('!I', transport.MAX_HEADER_BYTES + 1))
        with self.assertRaises(ValueError):
            transport.receive_tensors(connection)

    def test_client_reconnects_after_a_failed_connection(self):
        with patch.object(transport.socket, 'socket') as socket_factory:
            socket_factory.return_value.connect.side_effect = IOError('missing socket')
            client = transport.InferenceClient('/unused/inference.sock', timeout=0.1)
            with self.assertRaises(OSError):
                client.infer({'speed': np.ones(1, dtype=np.float32)})
            self.assertIsNone(client.connection)
            socket_factory.return_value.close.assert_called_once()

    @unittest.skipUnless(os.environ.get('XMOBILITY_RUN_SOCKET_SERVER_TEST') == '1',
                         'Set XMOBILITY_RUN_SOCKET_SERVER_TEST=1 to test socket bind/listen')
    def test_client_and_service_exchange_multiple_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/inference.sock'
            process = multiprocessing.Process(target=server.serve, args=(EchoRunner(), path))
            process.start()
            client = transport.InferenceClient(path, timeout=2)
            try:
                deadline = time.monotonic() + 3
                while not Path(path).exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(Path(path).exists())
                for value in (2, 7):
                    result = client.infer({'speed': np.array([value], dtype=np.float32)})
                    np.testing.assert_array_equal(result['speed_reply'], [value + 1])
                self.assertIsNotNone(client.connection)
            finally:
                client.close()
                process.terminate()
                process.join(timeout=3)
                self.assertFalse(process.is_alive())

    def test_invalid_client_timeout_is_rejected(self):
        for timeout in (0, -1, float('nan'), float('inf')):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                transport.InferenceClient('/tmp/unused.sock', timeout)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.shapes = {
            'image': (1, 1, 3, 1, 2), 'route': (1, 1, 19, 4), 'speed': (1, 1, 1),
            'action_input': (1, 6), 'history_input': (1, 1024), 'sample_input': (1, 512),
            'action_output': (1, 6), 'path_output': (1, 10),
            'history_output': (1, 1024), 'sample_output': (1, 512),
        }
        # Deliberately shuffle bindings to detect assumptions about export order.
        self.names = ['history_output', 'speed', 'action_input', 'image', 'sample_output',
                      'route', 'history_input', 'sample_input', 'path_output', 'action_output']
        self.engine = Mock(num_bindings=10)
        self.engine.get_binding_name.side_effect = lambda index: self.names[index]
        self.engine.get_binding_shape.side_effect = lambda index: self.shapes[self.names[index]]
        self.engine.binding_is_input.side_effect = lambda index: self.names[index] in server.INPUT_NAMES
        self.engine.get_binding_dtype.return_value = 'float32'
        self.cuda = ModuleType('pycuda.driver')
        self.cuda.mem_alloc = Mock(side_effect=range(1, 11))
        self.cuda.memcpy_htod = Mock()
        self.cuda.memcpy_dtoh = Mock(side_effect=lambda dest, source: dest.fill(source))
        self.trt = ModuleType('tensorrt')
        self.trt.__version__ = '8.6.1'
        self.trt.Logger = Mock()
        self.trt.init_libnvinfer_plugins = Mock()
        self.trt.nptype = np.dtype
        self.trt.Runtime = Mock()
        self.trt.Runtime.return_value.deserialize_cuda_engine.return_value = self.engine
        self.modules = {'pycuda': ModuleType('pycuda'), 'pycuda.driver': self.cuda,
                        'pycuda.autoinit': ModuleType('pycuda.autoinit'), 'tensorrt': self.trt}

    def runner(self):
        with tempfile.NamedTemporaryFile() as engine_file:
            engine_file.write(b'engine')
            engine_file.flush()
            with patch.dict(sys.modules, self.modules):
                return server.TensorRTRunner(engine_file.name)

    def inputs(self):
        return {name: np.ones(self.shapes[name], dtype=np.float32)
                for name in server.INPUT_NAMES}

    def test_binding_order_is_read_from_engine_and_buffers_are_reused(self):
        runner = self.runner()
        for _ in range(2):
            result = runner.infer(self.inputs())
            self.assertEqual(set(result), set(server.OUTPUT_SIZES))
            np.testing.assert_array_equal(result['action_output'], np.full((1, 6), 10))
        self.assertEqual(self.cuda.mem_alloc.call_count, 10)
        context = self.engine.create_execution_context.return_value
        context.execute_v2.assert_called_with(list(range(1, 11)))

    def test_wrong_image_size_is_rejected_before_execution(self):
        runner = self.runner()
        inputs = self.inputs()
        inputs['image'] = np.ones((3, 2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'image'):
            runner.infer(inputs)
        self.engine.create_execution_context.return_value.execute_v2.assert_not_called()

    def test_extra_outputs_are_rejected_at_startup(self):
        self.names[-1] = 'semantic_output'
        self.shapes['semantic_output'] = (1, 6)
        with self.assertRaisesRegex(ValueError, 'six inputs and four outputs'):
            self.runner()

    def test_unsupported_tensorrt_version_is_rejected(self):
        self.trt.__version__ = '10.1.0'
        with self.assertRaisesRegex(RuntimeError, 'TensorRT 8'):
            self.runner()

    def test_failed_engine_deserialization_is_reported(self):
        self.trt.Runtime.return_value.deserialize_cuda_engine.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'deserialize'):
            self.runner()

    def test_dynamic_engine_shapes_are_rejected(self):
        self.shapes['image'] = (1, 1, 3, -1, -1)
        with self.assertRaisesRegex(ValueError, 'fixed'):
            self.runner()


if __name__ == '__main__':
    unittest.main()
