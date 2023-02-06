from phaedra.sources import Source


class Web(Source):
    @staticmethod
    def extract_text(url: str) -> str:
        raise NotImplementedError
