
"""Reference hardware profiles for heterogeneous federated learning clients.

Specifications describe real reference devices.
They are metadata only: they do not allocate or limit hardware resources.
"""

CLIENT_PROFILES = {
    0: {
        "device": "Raspberry Pi 4 Model B",
        "cpu": {
            "architecture": "ARM Cortex-A72",
            "cores": 4,
            "frequency_ghz": 1.8,
        },
        "ram_gb": 4,
        "gpu": {
            "architecture": "VideoCore VI",
            "cuda": False,
            "cuda_cores": None,
            "tensor_cores": None,
            "ai_tops": None,
        },
        "network": {
            "ethernet_mbps": 1000,
            "wifi": "802.11ac",
        },
    },

    1: {
        "device": "Raspberry Pi 5",
        "cpu": {
            "architecture": "ARM Cortex-A76",
            "cores": 4,
            "frequency_ghz": 2.4,
        },
        "ram_gb": 8,
        "gpu": {
            "architecture": "VideoCore VII",
            "cuda": False,
            "cuda_cores": None,
            "tensor_cores": None,
            "ai_tops": None,
        },
        "network": {
            "ethernet_mbps": 1000,
            "wifi": "802.11ac",
        },
    },

    2: {
        "device": "NVIDIA Jetson Nano 4GB",
        "cpu": {
            "architecture": "ARM Cortex-A57",
            "cores": 4,
            "frequency_ghz": None,
        },
        "ram_gb": 4,
        "gpu": {
            "architecture": "NVIDIA Maxwell",
            "cuda": True,
            "cuda_cores": 128,
            "tensor_cores": None,
            "ai_tops": None,
        },
        "network": {
            "ethernet_mbps": 1000,
            "wifi": None,
        },
    },

    3: {
        "device": "NVIDIA Jetson Xavier NX 8GB",
        "cpu": {
            "architecture": "NVIDIA Carmel ARMv8.2",
            "cores": 6,
            "frequency_ghz": 1.9,
        },
        "ram_gb": 8,
        "gpu": {
            "architecture": "NVIDIA Volta",
            "cuda": True,
            "cuda_cores": 384,
            "tensor_cores": 48,
            "ai_tops": 21,
        },
        "network": {
            "ethernet_mbps": 1000,
            "wifi": None,
        },
    },

    4: {
        "device": "NVIDIA Jetson Orin Nano Super 8GB",
        "cpu": {
            "architecture": "ARM Cortex-A78AE",
            "cores": 6,
            "frequency_ghz": None,
        },
        "ram_gb": 8,
        "gpu": {
            "architecture": "NVIDIA Ampere",
            "cuda": True,
            "cuda_cores": 1024,
            "tensor_cores": 32,
            "ai_tops": 67,
        },
        "network": {
            "ethernet_mbps": 1000,
            "wifi": None,
        },
    },
}
