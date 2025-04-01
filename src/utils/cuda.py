import torch


def get_device() -> torch.device:
    """
    Get the device to use for PyTorch operations.
    Returns:
        torch.device: The device to use (CPU or GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_cuda():
    """
    Check if CUDA is available and print the device name.
    Raises:
        RuntimeError: If CUDA is not available.
    """
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Please check your installation.")
