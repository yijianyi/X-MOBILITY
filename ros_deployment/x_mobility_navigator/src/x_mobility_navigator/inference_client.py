"""Float32 tensor transport shared by Python 2.7 ROS and Python 3 inference."""

from __future__ import absolute_import

import json
import socket
import struct

import numpy as np

MAX_HEADER_BYTES = 65536
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_TENSORS = 16


def receive_exact(connection, length):
    data = bytearray(length)
    view = memoryview(data)
    offset = 0
    while offset < length:
        count = connection.recv_into(view[offset:], length - offset)
        if count == 0:
            raise IOError('Inference connection closed')
        offset += count
    return data


def send_tensors(connection, tensors):
    arrays = [(name, np.ascontiguousarray(value, dtype='<f4'))
              for name, value in sorted(tensors.items())]
    # ascontiguousarray promotes scalar inputs to shape (1,), which carries
    # the same float32 speed value across Python versions.
    metadata = [dict(name=name, shape=list(value.shape), nbytes=value.nbytes)
                for name, value in arrays]
    encoded = json.dumps(metadata).encode('utf-8')
    if (not 0 < len(arrays) <= MAX_TENSORS
            or len(encoded) > MAX_HEADER_BYTES
            or sum(value.nbytes for _, value in arrays) > MAX_PAYLOAD_BYTES):
        raise ValueError('Inference request is too large or empty')
    connection.sendall(struct.pack('!I', len(encoded)))
    connection.sendall(encoded)
    for _, value in arrays:
        connection.sendall(value.tobytes())


def receive_tensors(connection):
    header_size = struct.unpack('!I', receive_exact(connection, 4))[0]
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise ValueError('Invalid inference header size')
    metadata = json.loads(receive_exact(connection, header_size).decode('utf-8'))
    if not isinstance(metadata, list) or not 0 < len(metadata) <= MAX_TENSORS:
        raise ValueError('Invalid inference tensor list')
    specs = []
    names = set()
    total = 0
    for item in metadata:
        name, shape, nbytes = item['name'], item['shape'], item['nbytes']
        if (not isinstance(shape, list) or len(shape) > 8
                or not name or name in names):
            raise ValueError('Invalid inference tensor metadata')
        count = 1
        for dimension in shape:
            if type(dimension) is not int or dimension <= 0:
                raise ValueError('Invalid inference tensor shape')
            count *= dimension
        if nbytes != count * 4:
            raise ValueError('Inference tensor shape and size disagree')
        total += nbytes
        if total > MAX_PAYLOAD_BYTES:
            raise ValueError('Inference payload exceeds size limit')
        names.add(name)
        specs.append((name, shape, nbytes))
    return {
        name: np.frombuffer(receive_exact(connection, nbytes), dtype='<f4').reshape(shape)
        for name, shape, nbytes in specs
    }


class InferenceClient(object):
    def __init__(self, socket_path, timeout=5.0):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('inference_timeout must be positive and finite')
        self.socket_path = socket_path
        self.timeout = timeout
        self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def infer(self, inputs):
        try:
            if self.connection is None:
                self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.connection.settimeout(self.timeout)
                self.connection.connect(self.socket_path)
            send_tensors(self.connection, inputs)
            return receive_tensors(self.connection)
        except (socket.error, IOError, OSError, ValueError, KeyError, TypeError):
            self.close()
            raise
