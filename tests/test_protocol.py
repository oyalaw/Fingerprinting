import socket
import threading

import numpy as np

from ai_fingerprint.protocol import (
    array_to_bytes,
    arrays_to_bytes,
    bytes_to_array,
    bytes_to_arrays,
    bytes_to_training_batch,
    recv_frame,
    send_frame,
    training_batch_to_bytes,
)


def test_frame_round_trip():
    left, right = socket.socketpair()
    payload = array_to_bytes(np.arange(12).reshape(3, 4))

    def sender():
        send_frame(
            left,
            {
                "op": "infer",
                "request_id": "abc",
            },
            payload,
        )
        left.close()

    thread = threading.Thread(target=sender)
    thread.start()

    header, received = recv_frame(right)
    right.close()
    thread.join()

    assert header["op"] == "infer"
    assert np.array_equal(
        bytes_to_array(received),
        np.arange(12).reshape(3, 4),
    )



def test_multi_array_round_trip():
    arrays = [
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.asarray([1, 2], dtype=np.int64),
    ]
    payload = arrays_to_bytes(arrays)
    restored = bytes_to_arrays(payload)

    assert len(restored) == 2
    assert np.array_equal(restored[0], arrays[0])
    assert np.array_equal(restored[1], arrays[1])


def test_training_batch_round_trip():
    inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    targets = np.asarray([1, 0], dtype=np.int64)
    payload = training_batch_to_bytes(inputs, targets)
    restored_inputs, restored_targets = bytes_to_training_batch(payload)

    assert np.array_equal(restored_inputs, inputs)
    assert np.array_equal(restored_targets, targets)
