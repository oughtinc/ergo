from abc import ABC, abstractmethod


class Question(ABC):
    """A question from a forecasting platform"""

    @abstractmethod
    def get_text(self):
        """
        Get the summarizing text of the question
        """
        raise NotImplementedError("This should be implemented by a subclass")

    @abstractmethod
    def refresh(self):
        """
        Retrieves the latest information on the question from the prediction platform
        """
        raise NotImplementedError("This should be implemented by a subclass")

    @abstractmethod
    def sample_community(self):
        """
        Get one sample from the distribution of the community's
        prediction on this question
        (sample is denormalized/on the the true scale of the question)
        """
        raise NotImplementedError("This should be implemented by a subclass")
