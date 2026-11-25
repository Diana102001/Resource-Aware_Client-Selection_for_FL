# from collections import Counter
# from flwr_datasets import FederatedDataset
# from pytorchexample.task import (
#     DATASET_NAME,
#     create_partitioner,
#     shuffle_train_split,
# )

# fds = FederatedDataset(
#     dataset=DATASET_NAME,
#     partitioners={"train": create_partitioner(5)},
#     shuffle=False,
#     preprocessor=shuffle_train_split,
# )

# for i in range(5):
#     partition = fds.load_partition(i)
#     counts = dict(sorted(Counter(partition["label"]).items()))

#     print(f"\nClient {i}")
#     print(f"Total samples: {len(partition)}")
#     print(f"Class distribution: {counts}")
from pytorchexample.resources import CLIENT_PROFILES

for client_id, profile in CLIENT_PROFILES.items():
    print(f"Client {client_id}: {profile['device']}")
    print(f"  CPU: {profile['cpu']['architecture']}")
    print(f"  RAM: {profile['ram_gb']} GB")
    print(f"  GPU: {profile['gpu']['architecture']}")
    print()