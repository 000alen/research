import abc


class SourceInterface(abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def extract_text(*args, **kwargs) -> str:
        raise NotImplementedError
