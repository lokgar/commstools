import abc
from .signal import Signal

class ProcessingBlock(abc.ABC):
    """
    Abstract base class for signal processing blocks.
    
    All processing blocks must implement the `process` method, which takes a 
    Signal object and returns a processed Signal object.
    """
    
    @abc.abstractmethod
    def process(self, signal: Signal) -> Signal:
        """
        Process the input signal and return the result.
        
        Args:
            signal: The input Signal object.
            
        Returns:
            The processed Signal object.
        """
        pass
        
    def __call__(self, signal: Signal) -> Signal:
        """
        Allows the block to be called like a function.
        """
        return self.process(signal)
