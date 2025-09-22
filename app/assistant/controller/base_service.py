from abc import ABC, abstractmethod

class BaseService(ABC):
    """Abstract base class for model services"""
    
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        pass