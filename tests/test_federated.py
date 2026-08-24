import numpy as np

from ai_fingerprint.federated import ClientUpdate, fedavg


def test_fedavg_weighted_average():
    first = ClientUpdate(
        parameters=[
            np.asarray([1.0, 3.0], dtype=np.float32),
            np.asarray([2], dtype=np.int64),
        ],
        num_examples=1,
        metrics={},
    )
    second = ClientUpdate(
        parameters=[
            np.asarray([5.0, 7.0], dtype=np.float32),
            np.asarray([4], dtype=np.int64),
        ],
        num_examples=3,
        metrics={},
    )

    result = fedavg([first, second])

    assert np.allclose(result[0], np.asarray([4.0, 6.0], dtype=np.float32))
    assert result[1].dtype == np.int64
    assert int(result[1][0]) == 4
