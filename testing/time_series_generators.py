from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


class TimeSeriesScalerGenerator(Sequence):
    def __init__(self, data, target, features, look_back, batch_size, **kwargs):
        """
        Time Series Data Generator with Standard Scaling Per Sample.

        Args:
            data (pd.DataFrame): DataFrame with feature columns.
            target (str): Target column name.
            features (list): List of all feature column names.
            look_back (int): Number of past time steps per sample.
            batch_size (int): Batch size.
        """
        super().__init__(**kwargs)

        self.data = data[features].values  # Feature matrix
        self.targets = data[target].values.astype(int)  # Target labels
        self.features = features
        self.look_back = look_back
        self.batch_size = batch_size

        # Compute valid indices for sequence extraction
        self.indices = np.arange(len(data) - look_back)

    def __len__(self):
        """Returns number of batches per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        """Generates one batch of data."""
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        # Extract sequences
        batch_data = np.array(
            [self.data[i : i + self.look_back] for i in batch_indices]
        )

        # Allocate space for scaled batch
        X_batch = np.empty_like(batch_data, dtype=np.float32)
        y_batch = np.array(
            [self.targets[i + self.look_back] for i in batch_indices], dtype=np.int32
        )

        # Standardize each sequence individually
        for i, seq in enumerate(batch_data):
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_batch[i] = scaler.fit_transform(seq)

        return X_batch, y_batch


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

        # Store true labels for the entire dataset
        self.true_labels = np.array(
            [self.targets[i + self.look_back] for i in self.indices]
        )

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
        y_batch = self.true_labels[batch_indices]

        return X_batch, y_batch
