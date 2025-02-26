from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# # Format for scaling indices definition
# standard_indices = [features.index(f) for f in standard_features]
# minmax_indices = [features.index(f) for f in minmax_features]
# unscaled_indices = [features.index(f) for f in unscaled_features]


class TimeSeriesScalerGenerator(Sequence):
    def __init__(
        self,
        data,
        target,
        features,
        standard_features,
        minmax_features,
        unscaled_features,
        look_back,
        batch_size=32,
        **kwargs
    ):
        """
        Custom Timeseries Generator with pre-scaled data.

        Args:
            data (pd.DataFrame): DataFrame with feature columns.
            target (str): Target column name.
            features (list): List of all feature column names.
            standard_features (list): Features to be standardized.
            minmax_features (list): Features to be Min-Max scaled.
            unscaled_features (list): Features to remain unscaled.
            look_back (int): Number of past time steps per sample.
            batch_size (int): Batch size.
        """
        super().__init__(**kwargs)

        self.data = data[features].values  # Extract feature matrix
        self.targets = data[target].values.astype(int)  # Extract target labels
        self.features = features
        self.standard_features = standard_features
        self.minmax_features = minmax_features
        self.unscaled_features = unscaled_features
        self.look_back = look_back
        self.batch_size = batch_size

        self.standard_indices = [features.index(f) for f in standard_features]
        self.minmax_indices = [features.index(f) for f in minmax_features]
        self.unscaled_indices = [features.index(f) for f in unscaled_features]

        self.indices = np.arange(len(data) - look_back)
        self.true_labels = self._extract_true_labels()

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        """Generates one batch of data."""
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        # Extract sequences efficiently using list slicing
        batch_data = np.array(
            [self.data[i : i + self.look_back] for i in batch_indices]
        )

        # Preallocate arrays for batch
        X_batch = np.empty(
            (len(batch_indices), self.look_back, len(self.features)), dtype=np.float32
        )
        y_batch = np.empty(len(batch_indices), dtype=np.int32)

        # Scale using precomputed scalers
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler(feature_range=(0, 1))

        for i, seq in enumerate(batch_data):
            seq_standard = standard_scaler.fit_transform(seq[:, self.standard_indices])
            seq_minmax = minmax_scaler.fit_transform(seq[:, self.minmax_indices])
            seq_unscaled = (
                seq[:, self.unscaled_indices]
                if self.unscaled_features
                else np.empty((self.look_back, 0))
            )

            X_batch[i] = np.hstack((seq_standard, seq_minmax, seq_unscaled))
            y_batch[i] = self.targets[batch_indices[i] + self.look_back]

        return X_batch, y_batch

    def _extract_true_labels(self):
        """Extract all true labels for the entire dataset."""
        return np.array([self.targets[i + self.look_back] for i in self.indices])


class TimeSeriesGenerator(Sequence):
    def __init__(self, scaled_data, target, look_back, batch_size=32, **kwargs):
        """
        Custom TimeSeries Generator with pre-scaled data.

        Args:
            scaled_data (np.ndarray): Pre-scaled feature matrix.
            target (np.ndarray): Target values.
            look_back (int): Number of past time steps per sample.
            batch_size (int): Batch size.
        """
        super().__init__(**kwargs)

        self.data = scaled_data  # Already scaled data
        self.targets = target.astype(int)  # Target labels
        self.look_back = look_back
        self.batch_size = batch_size
        self.indices = np.arange(len(scaled_data) - look_back)

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        """Generates one batch of data."""
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        # Extract sequences efficiently
        X_batch = np.array([self.data[i : i + self.look_back] for i in batch_indices])
        y_batch = np.array([self.targets[i + self.look_back] for i in batch_indices])

        return X_batch, y_batch
