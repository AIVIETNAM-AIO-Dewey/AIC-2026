"""SigLIP2 model identity used by the CPU worker contract."""

MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DIMENSION = 768
MAX_TOKENS = 64

__all__ = ["DIMENSION", "MAX_TOKENS", "MODEL_ID", "REVISION"]
