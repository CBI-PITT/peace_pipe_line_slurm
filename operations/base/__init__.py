class ImageOperation:
    def __init__(self, input, output, **kwargs):
        self.input = input
        self.output = output
        self.params = kwargs

    def run(self):
        """Perform the operation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


class ImageReader:
    def __init__(self, input, output, **kwargs):
        self.input = input
        self.output = output
        self.params = kwargs

    def run(self):
        """Perform the reading. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
