from abc import ABC, abstractmethod
import pickle

class Persistable(ABC):

    @staticmethod
    def from_file(filename: str) -> object:
        with open(filename, "rb") as load_input:
            loaded_object = pickle.load(load_input)
        return loaded_object

    def to_file(self, filename: str):
        with open(filename, "wb") as save_output:
            pickle.dump(self, save_output, pickle.HIGHEST_PROTOCOL)