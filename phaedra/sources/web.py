from phaedra.sources import SourceInterface


class Web(SourceInterface):
    @staticmethod
    def extract_text(url: str) -> str:
        raise NotImplementedError
