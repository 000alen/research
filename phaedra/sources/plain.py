from phaedra.sources import SourceInterface


class Plain(SourceInterface):
    @staticmethod
    def extract_text(path: str) -> str:
        with open(path, "r") as f:
            return f.read()
