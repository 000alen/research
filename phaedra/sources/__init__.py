import abc


class Source(abc.ABCMeta):
    # origin: str
    # type: str
    # metadata: dict
    # content: list[str]

    # @classmethod
    # def from_dict(cls, *args, **kwargs):
    #     raise NotImplementedError

    # def to_dict(self, *args, **kwargs):
    #     raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def extract_text(*args, **kwargs) -> str:
        raise NotImplementedError
