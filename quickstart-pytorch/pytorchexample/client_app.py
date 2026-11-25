
"""EuroSAT RGB: Flower federated learning client."""

import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import (
    MobileNetV2,
    load_data,
    set_seed,
    test as test_fn,
    train as train_fn,
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    # Client information
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    print(
        f"\n[TRAINING] Client {partition_id} selected!",
        flush=True,
    )

    # Reproducibility
    set_seed(42 + partition_id)

    # Initialize model and load global parameters
    model = MobileNetV2()
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Load local training data
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
    )

    num_examples = len(trainloader.dataset)
    num_batches = len(trainloader)

    # Measure actual local training time
    start_time = time.perf_counter()

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    actual_training_time = time.perf_counter() - start_time

    print(
        f"[CLIENT {partition_id}] "
        f"Time: {actual_training_time:.3f}s | "
        f"Loss: {train_loss:.4f} | "
        f"Samples: {num_examples} | "
        f"Batches: {num_batches}",
        flush=True,
    )

    # Prepare model parameters
    model_record = ArrayRecord(model.state_dict())

    # Return metrics to the server
    metrics = {
        "partition_id": partition_id,
        "train_loss": train_loss,
        "num-examples": num_examples,
        "num-batches": num_batches,
        "actual_training_time": actual_training_time,
    }

    metric_record = MetricRecord(metrics)

    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record,
    })

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Client-side evaluation (disabled in the current experiment)."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    set_seed(42 + partition_id)

    model = MobileNetV2()
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
    )

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }

    content = RecordDict({
        "metrics": MetricRecord(metrics),
    })

    return Message(content=content, reply_to=msg)
