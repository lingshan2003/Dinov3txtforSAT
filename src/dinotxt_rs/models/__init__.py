from .embedding_adapter import DINOtxtWithImageAdapter, add_image_embedding_adapter
from .official_dinotxt import configure_trainable_parameters, load_official_dinotxt

__all__ = [
    "DINOtxtWithImageAdapter",
    "add_image_embedding_adapter",
    "configure_trainable_parameters",
    "load_official_dinotxt",
]
