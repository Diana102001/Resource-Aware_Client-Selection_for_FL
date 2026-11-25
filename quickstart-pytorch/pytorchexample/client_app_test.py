
"""Lightweight client app for testing random client selection."""

from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Report participation without performing model training."""

    partition_id = context.node_config["partition-id"]

    print(f"\n[SELECTION TEST] Client {partition_id} was selected!")

    metrics = MetricRecord({
        "partition_id": partition_id,
        "num-examples": 1,
    })

    return Message(
        content=RecordDict({
            "arrays": msg.content["arrays"],
            "metrics": metrics,
        }),
        reply_to=msg,
    )
