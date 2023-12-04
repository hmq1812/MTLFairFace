import torch

class OGDMemory:
    """
    Class to manage memory for Orthogonal Gradient Descent (OGD).
    It stores gradients or model states necessary for OGD.
    """

    def __init__(self, memory_size):
        """
        Initializes the OGDMemory object.
        Args:
            memory_size (int): The maximum size of the memory.
        """
        self.memory_size = memory_size
        self.memory = []

    def add(self, item):
        """
        Adds an item to the memory.
        Args:
            item: The item to be added.
        """
        if len(self.memory) < self.memory_size:
            self.memory.append(item)
        else:
            self.memory.pop(0)
            self.memory.append(item)

    def get(self):
        """
        Retrieves all items in the memory.
        Returns:
            List: The list of items in the memory.
        """
        return self.memory

    def clear(self):
        """
        Clears the memory.
        """
        self.memory.clear()

    def __len__(self):
        """
        Returns the current size of the memory.
        """
        return len(self.memory)
