
"""Lightweight server app for testing random client selection."""

import torch

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run selection tests without training a real model."""

    
    strategy = FedAvg(
        fraction_train=0.6,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=3,
        min_available_nodes=5,
    )


    # A single dummy parameter instead of MobileNetV2
    initial_arrays = ArrayRecord({
        "dummy_weight": torch.tensor([0.0])
    })

    print("\nStarting random client-selection test...")

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({}),
        num_rounds=3,
    )

    print("\nSelection test completed.")
