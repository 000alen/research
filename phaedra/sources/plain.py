from phaedra.sources import Source


class Plain(Source):
    @staticmethod
    def extract_text(path: str) -> str:
        with open(path, "r") as f:
            return f.read()
