
"""Shared tasks and dataset utilities for EuroSAT federated learning."""

import random

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, Normalize, ToTensor


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


# Dataset
DATASET_NAME = "blanchon/EuroSAT_RGB"
NUM_CLASSES = 10
IMAGE_SIZE = 64  # Native EuroSAT RGB image dimensions

# Federated clients
NUM_PARTITIONS = 5
BASE_SEED = 42

# Data partitioning
DIRICHLET_ALPHA = 0.1
MIN_PARTITION_SIZE = 500
LOCAL_VALIDATION_SIZE = 0.2

# Model
MODEL_NAME = "MobileNetV2"
MODEL_WEIGHTS = None  # Train from scratch

# Image preprocessing
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Local training
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Optimizer
OPTIMIZER_NAME = "SGD"
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0

# Loss function
LOSS_FUNCTION = "CrossEntropyLoss"

# Learning-rate scheduler
SCHEDULER_NAME = None
LEARNING_RATE_DECAY = None

# Federated learning
NUM_SERVER_ROUNDS = 30
FRACTION_TRAIN = 0.6
FRACTION_EVALUATE = 0.0
MIN_TRAIN_NODES = 3
MIN_AVAILABLE_NODES = 5

# Aggregation
AGGREGATION_STRATEGY = "FedAvg"
AGGREGATION_WEIGHTED_BY = "num-examples"

# Evaluation
CENTRALIZED_VALIDATION = True
CLIENT_SIDE_EVALUATION = False
FINAL_TEST_EVALUATION = True

# ---------------------------------------------------------------------
# Dataset caching
# ---------------------------------------------------------------------

_dataset_splits = None


def get_dataset_splits():
    """Load and cache the EuroSAT dataset splits."""

    global _dataset_splits

    if _dataset_splits is None:
        print("Loading EuroSAT RGB dataset...", flush=True)

        _dataset_splits = load_dataset(DATASET_NAME)

        # Shuffle the training split before partitioning.
        _dataset_splits["train"] = _dataset_splits["train"].shuffle(
            seed=BASE_SEED,
            keep_in_memory=True,
        )

        print("EuroSAT dataset loaded successfully.", flush=True)

    return _dataset_splits


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed=BASE_SEED):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def MobileNetV2():
    """Create MobileNetV2 with a randomly initialized 10-class classifier."""

    model = mobilenet_v2(weights=MODEL_WEIGHTS)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model


# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

transform = Compose([
    ToTensor(),
    Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD,
    ),
])


def apply_transforms(batch):
    """Convert PIL images into normalized PyTorch tensors."""

    batch["image"] = [
        transform(image.convert("RGB"))
        for image in batch["image"]
    ]

    return batch


# ---------------------------------------------------------------------
# Federated data partitioning
# ---------------------------------------------------------------------

def create_partitioner():
    """Create the Dirichlet partitioner used by all clients."""

    return DirichletPartitioner(
        num_partitions=NUM_PARTITIONS,
        partition_by="label",
        alpha=DIRICHLET_ALPHA,
        min_partition_size=MIN_PARTITION_SIZE,
        self_balancing=False,
        shuffle=True,
        seed=BASE_SEED,
    )


def load_data(partition_id, num_partitions, batch_size):
    """
    Load one client's data and split it into local training
    and validation subsets.

    The partitioning is non-IID and uses a Dirichlet distribution.
    """

    if num_partitions != NUM_PARTITIONS:
        raise ValueError(
            f"Expected {NUM_PARTITIONS} partitions, "
            f"but received {num_partitions}."
        )

    if not 0 <= partition_id < num_partitions:
        raise ValueError(f"Invalid partition ID: {partition_id}")

    dataset_splits = get_dataset_splits()
    train_dataset = dataset_splits["train"]

    # Create the partitioner and assign the shuffled training dataset.
    partitioner = create_partitioner()
    partitioner.dataset = train_dataset

    # Retrieve this client's partition.
    partition = partitioner.load_partition(partition_id)

    # Split the client's data into local training and validation sets.
    partition = partition.train_test_split(
        test_size=LOCAL_VALIDATION_SIZE,
        seed=BASE_SEED,
    )

    train_partition = partition["train"].with_transform(
        apply_transforms
    )

    validation_partition = partition["test"].with_transform(
        apply_transforms
    )

    # Give each client a reproducible but different shuffle sequence.
    generator = torch.Generator()
    generator.manual_seed(BASE_SEED + partition_id)

    trainloader = DataLoader(
        train_partition,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    validationloader = DataLoader(
        validation_partition,
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, validationloader


# ---------------------------------------------------------------------
# Centralized validation and test datasets
# ---------------------------------------------------------------------

def load_centralized_dataset(batch_size=32):
    """
    Load the complete centralized validation split.

    This dataset is used by the server to evaluate the aggregated
    global model after each communication round.
    """

    dataset_splits = get_dataset_splits()

    validation_dataset = dataset_splits["validation"].with_transform(
        apply_transforms
    )

    validationloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return validationloader


def load_test_dataset(batch_size=32):
    """
    Load the separate test split.

    Use this dataset for the final evaluation after training,
    not for selecting or tuning models during training.
    """

    dataset_splits = get_dataset_splits()

    test_dataset = dataset_splits["test"].with_transform(
        apply_transforms
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return testloader


# ---------------------------------------------------------------------
# Local model training
# ---------------------------------------------------------------------

def train(net, trainloader, epochs, lr, device):
    """
    Train a model on one client's local dataset.

    Uses the fixed SGD and CrossEntropyLoss configuration.
    Returns sample-weighted average training loss.
    """

    net.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    total_loss = 0.0
    total_samples = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            current_batch_size = labels.size(0)

            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

    return total_loss / total_samples


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------

def test(net, testloader, device):
    """
    Evaluate a model on a dataset.

    Returns sample-weighted average loss and accuracy.
    """

    net.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return average_loss, accuracy
