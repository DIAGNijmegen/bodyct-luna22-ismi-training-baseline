from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator, Iterator


def _sanitize_class_balance(
    classes: List[int], class_balance: Optional[Dict[int, float]] = None
) -> Dict[int, float]:
    if class_balance is None:
        output = {c: 1.0 / len(classes) for c in classes}
    else:
        # normalize the class fractions
        total = np.sum(list(class_balance.values()))
        output = {c: v / total for c, v in class_balance.items()}
    assert np.sum(list(output.values())) == 1.0
    return output


def sample_balanced(
    input_labels: np.ndarray,
    required_samples: int,
    class_balance: Optional[
        Dict[int, float]
    ] = None,  # by default sample classes equally
    shuffle: bool = True,
) -> Dict[int, List[int]]:
    assert input_labels.ndim == 1
    # split input_labels
    classes, indices = np.unique(input_labels, return_index=True)
    classes = list(map(int, classes))
    class_balance = _sanitize_class_balance(
        classes=classes, class_balance=class_balance
    )
    index_dict = {
        c: np.where(input_labels == c)[0]
        if not shuffle
        else np.random.permutation(np.where(input_labels == c)[0])
        for c in classes
    }
    indices_per_class = {
        c: index_dict[c][: int(class_balance[c] * required_samples)] for c in classes
    }
    assert np.sum(list(map(len, indices_per_class.values()))) == required_samples
    return indices_per_class


def sample_balanced_batches(
    input_labels: np.ndarray,
    required_batches: int,
    batch_size: int,
    class_balance: Optional[Dict[int, float]] = None,
    shuffle: bool = True,
) -> np.ndarray:
    assert input_labels.ndim == 1

    indices_per_class = sample_balanced(
        input_labels=input_labels,
        required_samples=required_batches * batch_size,
        class_balance=class_balance,
        shuffle=shuffle,
    )
    classes = list(indices_per_class.keys())

    samples_per_batch = {c: int(class_balance[c] * batch_size) for c in classes}
    assert np.sum(list(samples_per_batch.values())) == batch_size

    sampled = []
    for i in range(required_batches):
        batch = []
        for c in classes:
            samples = samples_per_batch[c]
            batch.append(indices_per_class[c][samples * i : samples * (i + 1)])
        sampled.append(np.concatenate(batch))

    return np.array(sampled)


class UndersamplingIterator(Iterator):
    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        class_balance: Optional[Dict[int, float]] = None,
        shuffle: bool = True,
        preprocess_fn: Callable = None,
        seed: np.random.RandomState = None,
    ):
        self._inputs = inputs
        self._labels = labels
        self._preprocess_fn = preprocess_fn
        self._labels_argmax = np.argmax(self._labels, axis=1)
        self._batch_size = batch_size
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._req_batches = self._compute_required_batches()
        self._seed = seed
        super(UndersamplingIterator, self).__init__(
            n=self._req_batches,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            seed=self._seed,
        )
        self._sampled_batch_indices = []
        self._apply_balance_strategy()

    def _compute_required_batches(self) -> int:
        classes, counts = np.unique(self._labels_argmax, return_counts=True)
        self._class_balance = _sanitize_class_balance(
            classes=classes, class_balance=self._class_balance
        )
        min_class = np.argmin(
            [counts[i] / self._class_balance[c] for i, c in enumerate(classes)]
        )
        required_batches = int(
            np.floor(
                int(counts[min_class] / self._class_balance[min_class])
                // self._batch_size
            )
        )
        assert required_batches > 0
        return required_batches

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self._apply_balance_strategy()

    def _apply_balance_strategy(self):
        self._sampled_batch_indices = sample_balanced_batches(
            input_labels=self._labels_argmax,
            required_batches=self._req_batches,
            batch_size=self._batch_size,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
        )

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gets batch at position `index`.
        Args:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        indices = self._sampled_batch_indices[index]
        X, y = self._inputs[indices, :], self._labels[indices, :]
        if self._preprocess_fn is not None:
            X = self._preprocess_fn(X)
        return X, y

    def __len__(self) -> int:
        return self._req_batches
