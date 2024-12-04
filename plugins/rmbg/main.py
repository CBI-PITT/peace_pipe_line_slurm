from operations.base import ImageOperation


class rmbg(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        print("Initialized rmbg")

    def run(self):
        print("Running rmbg")
