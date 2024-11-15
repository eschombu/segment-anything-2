import logging
import os
import torch

TRAINING_MODE_VAR = "SAM2_TRAINING_ENABLED"
TRUE = "true"
FALSE = "false"


def is_training_enabled() -> bool:
    return os.environ.get(TRAINING_MODE_VAR, FALSE).lower() == TRUE


def enable_training():
    logging.info("Enabling training mode")
    os.environ[TRAINING_MODE_VAR] = TRUE


def disable_training():
    logging.info("Disabling training mode")
    os.environ[TRAINING_MODE_VAR] = FALSE


class no_grad_if_not_training(torch.no_grad):
    def __enter__(self) -> None:
        if not is_training_enabled():
            super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not is_training_enabled():
            super().__exit__(exc_type, exc_val, exc_tb)


class inference_mode_if_not_training(torch.inference_mode):
    def __enter__(self) -> None:
        if not is_training_enabled():
            super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not is_training_enabled():
            super().__exit__(exc_type, exc_val, exc_tb)
