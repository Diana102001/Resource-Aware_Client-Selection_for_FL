
"""EuroSAT RGB: Flower server with random FedAvg experiment logging."""

import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
import flwr

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import (
    BASE_SEED,
    DATASET_NAME,
    DIRICHLET_ALPHA,
    MobileNetV2,
    load_centralized_dataset,
    load_test_dataset,
    set_seed,
    test,
)


# ---------------------------------------------------------------------
# Experiment configuration and result storage
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results/random_baseline")
MODELS_DIR = RESULTS_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Histories collected during training
validation_history = []
training_history = []
client_training_details = []
round_timing_history = []

# Best model according to centralized validation accuracy
best_validation_accuracy = -1.0
best_validation_round = None
best_model_state = None


# ---------------------------------------------------------------------
# CSV and JSON utilities
# ---------------------------------------------------------------------

def save_csv(rows, filepath, fieldnames):
    """Save a list of dictionaries to a CSV file."""

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(data, filepath):
    """Save a dictionary as a formatted JSON file."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_training_results():
    """Save training, client and round histories."""

    save_csv(
        training_history,
        RESULTS_DIR / "training_history.csv",
        [
            "round",
            "train_loss_weighted",
            "weighted_avg_client_training_time_seconds",
            "num_successful_clients",
        ],
    )

    save_csv(
        client_training_details,
        RESULTS_DIR / "client_training_details.csv",
        [
            "round",
            "partition_id",
            "node_id",
            "train_loss",
            "num_examples",
            "num_batches",
            "actual_training_time_seconds",
            "time_per_sample_ms",
            "time_per_batch_ms",
        ],
    )

    save_csv(
        round_timing_history,
        RESULTS_DIR / "round_timing.csv",
        [
            "round",
            "selected_node_ids",
            "successful_partition_ids",
            "num_successful_clients",
            "mean_client_training_time_seconds",
            "max_client_training_time_seconds",
            "min_client_training_time_seconds",
            "pre_aggregation_elapsed_seconds",
            "aggregation_seconds",
            "round_duration_seconds",
        ],
    )


def save_validation_results():
    """Save centralized validation history."""

    save_csv(
        validation_history,
        RESULTS_DIR / "validation_history.csv",
        [
            "round",
            "validation_loss",
            "validation_accuracy",
        ],
    )


# ---------------------------------------------------------------------
# Custom FedAvg strategy for experiment logging
# ---------------------------------------------------------------------

class RandomLoggingFedAvg(FedAvg):
    """
    Standard FedAvg with random client selection.

    This class does not change the selection algorithm.
    It only records the selected nodes and their training metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.round_start_times = {}
        self.selected_node_ids = {}

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Record the start time and selected node IDs."""

        start_time = time.perf_counter()

        messages = list(
            super().configure_train(
                server_round,
                arrays,
                config,
                grid,
            )
        )

        self.round_start_times[server_round] = start_time

        selected_ids = [
            msg.metadata.dst_node_id
            for msg in messages
        ]

        self.selected_node_ids[server_round] = selected_ids

        print(
            f"\n[ROUND {server_round}] "
            f"Selected server node IDs: {selected_ids}",
            flush=True,
        )

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Record individual client metrics, then perform normal FedAvg."""

        # Convert to a list because we need to inspect the replies
        # before passing them to the original FedAvg implementation.
        replies = list(replies)

        round_client_times = []
        successful_partition_ids = []

        for reply in replies:

            if reply.has_error():
                continue

            metrics = reply.content["metrics"]

            partition_id = int(metrics["partition_id"])
            node_id = reply.metadata.src_node_id

            train_loss = float(metrics["train_loss"])
            num_examples = int(metrics["num-examples"])
            num_batches = int(metrics["num-batches"])
            actual_training_time = float(
                metrics["actual_training_time"]
            )

            time_per_sample_ms = (
                actual_training_time / num_examples * 1000
                if num_examples > 0 else 0.0
            )

            time_per_batch_ms = (
                actual_training_time / num_batches * 1000
                if num_batches > 0 else 0.0
            )

            client_training_details.append({
                "round": server_round,
                "partition_id": partition_id,
                "node_id": node_id,
                "train_loss": train_loss,
                "num_examples": num_examples,
                "num_batches": num_batches,
                "actual_training_time_seconds": actual_training_time,
                "time_per_sample_ms": time_per_sample_ms,
                "time_per_batch_ms": time_per_batch_ms,
            })

            round_client_times.append(actual_training_time)
            successful_partition_ids.append(partition_id)

            print(
                f"[ROUND {server_round}] "
                f"Client {partition_id}: "
                f"{actual_training_time:.3f}s | "
                f"{num_examples} samples | "
                f"{num_batches} batches",
                flush=True,
            )

        # Perform the original Flower FedAvg aggregation.
        aggregation_start = time.perf_counter()

        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round,
            replies,
        )

        aggregation_seconds = (
            time.perf_counter() - aggregation_start
        )

        # Calculate elapsed time since clients were selected.
        round_start = self.round_start_times.get(server_round)

        if round_start is not None:
            round_duration = time.perf_counter() - round_start
        else:
            round_duration = aggregation_seconds

        pre_aggregation_seconds = max(
            0.0,
            round_duration - aggregation_seconds,
        )

        # Record aggregated metrics.
        if aggregated_metrics is not None:
            training_history.append({
                "round": server_round,
                "train_loss_weighted": aggregated_metrics.get(
                    "train_loss", ""
                ),
                "weighted_avg_client_training_time_seconds":
                    aggregated_metrics.get(
                        "actual_training_time", ""
                    ),
                "num_successful_clients": len(round_client_times),
            })

        # Record round timing.
        round_timing_history.append({
            "round": server_round,
            "selected_node_ids": ";".join(
                str(x)
                for x in self.selected_node_ids.get(server_round, [])
            ),
            "successful_partition_ids": ";".join(
                str(x)
                for x in successful_partition_ids
            ),
            "num_successful_clients": len(round_client_times),
            "mean_client_training_time_seconds": (
                sum(round_client_times) / len(round_client_times)
                if round_client_times else ""
            ),
            "max_client_training_time_seconds": (
                max(round_client_times)
                if round_client_times else ""
            ),
            "min_client_training_time_seconds": (
                min(round_client_times)
                if round_client_times else ""
            ),
            "pre_aggregation_elapsed_seconds": pre_aggregation_seconds,
            "aggregation_seconds": aggregation_seconds,
            "round_duration_seconds": round_duration,
        })

        # Save after every completed training round.
        # This protects the experiment history if the run is interrupted.
        save_training_results()

        print(
            f"[ROUND {server_round}] "
            f"Round duration: {round_duration:.3f}s | "
            f"Aggregation: {aggregation_seconds:.3f}s",
            flush=True,
        )

        return aggregated_arrays, aggregated_metrics


# ---------------------------------------------------------------------
# Flower ServerApp
# ---------------------------------------------------------------------

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run random-selection federated learning and save its results."""

    global best_validation_accuracy
    global best_validation_round
    global best_model_state

    # Read experiment configuration.
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]

    # Set random seeds.
    set_seed(BASE_SEED)

    # Initialize the global model.
    global_model = MobileNetV2()
    arrays = ArrayRecord(global_model.state_dict())

    # Save experiment configuration before training starts.
    experiment_config = {
        "experiment_name": "random_baseline",
        "selection_strategy": "random",
        "dataset": DATASET_NAME,
        "model": "MobileNetV2",
        "num_classes": 10,
        "num_clients": 5,
        "clients_selected_per_round": 3,
        "fraction_train": 0.6,
        "fraction_evaluate": fraction_evaluate,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": "SGD",
        "momentum": 0.9,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "seed": BASE_SEED,
        "centralized_validation": True,
        "test_evaluation": "Once after training",
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "flower_version": flwr.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "torch_num_threads": torch.get_num_threads(),
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
    }

    save_json(
        experiment_config,
        RESULTS_DIR / "experiment_config.json",
    )

    # Initialize standard FedAvg with random selection.
    strategy = RandomLoggingFedAvg(
        fraction_train=0.6,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=3,
        min_available_nodes=5,
    )

    print("\nStarting random-selection experiment...", flush=True)

    experiment_start = time.perf_counter()

    # Run federated training.
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    training_runtime = time.perf_counter() - experiment_start

    # Save final model.
    final_state_dict = result.arrays.to_torch_state_dict()

    torch.save(
        final_state_dict,
        MODELS_DIR / "final_model.pt",
    )

    # Save best model according to centralized validation accuracy.
    if best_model_state is not None:
        torch.save(
            best_model_state,
            MODELS_DIR / "best_model.pt",
        )

    # -----------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------

    print("\nEvaluating final model on the test dataset...", flush=True)

    final_model = MobileNetV2()
    final_model.load_state_dict(final_state_dict)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    final_model.to(device)

    test_dataloader = load_test_dataset(batch_size=batch_size)

    test_start = time.perf_counter()

    test_loss, test_accuracy = test(
        final_model,
        test_dataloader,
        device,
    )

    test_runtime = time.perf_counter() - test_start

    test_results = {
        "model_evaluated": "final_model",
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_accuracy_percentage": test_accuracy * 100,
        "num_test_samples": len(test_dataloader.dataset),
        "test_evaluation_time_seconds": test_runtime,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_accuracy_percentage":
            best_validation_accuracy * 100,
        "best_validation_round": best_validation_round,
    }

    save_json(
        test_results,
        RESULTS_DIR / "test_results.json",
    )

    # Save final experiment summary.
    summary = {
        "training_runtime_seconds": training_runtime,
        "training_runtime_minutes": training_runtime / 60,
        "test_evaluation_time_seconds": test_runtime,
        "final_validation_accuracy": (
            validation_history[-1]["validation_accuracy"]
            if validation_history else None
        ),
        "final_validation_loss": (
            validation_history[-1]["validation_loss"]
            if validation_history else None
        ),
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_round": best_validation_round,
        "final_test_accuracy": test_accuracy,
        "final_test_loss": test_loss,
        "completed_rounds": len(training_history),
        "experiment_end_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_json(
        summary,
        RESULTS_DIR / "experiment_summary.json",
    )

    print("\n" + "=" * 55)
    print("RANDOM BASELINE EXPERIMENT COMPLETED")
    print("=" * 55)
    print(f"Training time: {training_runtime / 60:.2f} minutes")
    print(f"Final validation accuracy: {summary['final_validation_accuracy']:.4f}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Best validation round: {best_validation_round}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Results saved in: {RESULTS_DIR.resolve()}")


# ---------------------------------------------------------------------
# Centralized validation
# ---------------------------------------------------------------------

def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
) -> MetricRecord:
    """
    Evaluate the aggregated global model on the centralized
    validation dataset after each round.
    """

    global best_validation_accuracy
    global best_validation_round
    global best_model_state

    model = MobileNetV2()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    validation_dataloader = load_centralized_dataset()

    val_loss, val_acc = test(
        model,
        validation_dataloader,
        device,
    )

    print(
        f"\n[VALIDATION] Round {server_round} | "
        f"Loss: {val_loss:.4f} | "
        f"Accuracy: {val_acc:.4f}",
        flush=True,
    )

    validation_history.append({
        "round": server_round,
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
    })

    # Keep the model with the highest validation accuracy.
    if val_acc > best_validation_accuracy:
        best_validation_accuracy = val_acc
        best_validation_round = server_round

        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        torch.save(
            best_model_state,
            MODELS_DIR / "best_model.pt",
        )

    # Save validation results after every evaluation.
    save_validation_results()

    return MetricRecord({
        "accuracy": val_acc,
        "loss": val_loss,
    })
