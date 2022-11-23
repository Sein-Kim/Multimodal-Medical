class BaseMoCoException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseMoCoException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseMoCoException):
    """Raised when the choice of dataset is invalid."""
