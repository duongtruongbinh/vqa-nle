from abc import ABC, abstractmethod
from .utils import extract_clean_model_name


class VQAModel(ABC):
    """Abstract base class for a VQA model."""

    def __init__(self, **kwargs):
        """
        Initializes the model and tokenizer/processor.
        """
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_path = None
        self.model_name = None

    def _set_clean_model_name(self):
        """Sets clean model name from model path."""
        if self.model_path:
            self.model_name = extract_clean_model_name(self.model_path)

    @abstractmethod
    def load_model(self):
        """
        Loads the model and processor/tokenizer from the specified path.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, question: str, image_path: str) -> tuple[str, str]:
        """
        Performs inference on a single image-question pair.

        Args:
            question (str): The visual question regarding the image.
            image_path (str): The file path to the input image.

        Returns:
            tuple[str, str]: The model's answer and explanation.
        """
        raise NotImplementedError 