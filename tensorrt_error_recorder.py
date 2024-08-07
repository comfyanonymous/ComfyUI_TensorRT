import logging
import threading

import tensorrt as trt

# TensorRT Python API Docs: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ErrorRecorder.html
#
# NOTE: TensorRT does not attempt to marshall errors across threads, so IErrorRecorder implementations must be thread-safe to allow for the possibility of errors occuring on multiple threads.
class TrTErrorRecorder(trt.IErrorRecorder):
    def __init__(self):
        self.lock = threading.Lock()
        self.error_list = []
        super().__init__()

    def clear(self):
        with self.lock:
            self.error_list = []

    def get_error_code(self, error_index):
        with self.lock:
            if error_index >= len(self.error_list) or error_index < 0:
                raise IndexError(f'Invalid error index "{error_index}"')

            return self.error_list[error_index]['error_code']

    def get_error_desc(self, error_index):
        with self.lock:
            if error_index >= len(self.error_list) or error_index < 0:
                raise IndexError(f'Invalid error index "{error_index}"')

            return self.error_list[error_index]['error_desc']

    def has_overflowed(self):
        return False

    def num_errors(self):
        with self.lock:
            return len(self.error_list)

    def report_error(self, error_code, error_desc):
        logging.error(f"TensorRT has encountered an error. ErrorCode: {error_code}. ErrorDesc: {error_desc}")

        with self.lock:
            self.error_list.append({'error_code': error_code, 'error_desc': error_desc})

        # TODO Future: return True for errors we consider 'fatal', which hints to TensorRT to stop execution.
        return False
