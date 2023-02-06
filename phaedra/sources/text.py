from phaedra.sources import Source


class Text(Source):
    @staticmethod
    def extract_text(text: str) -> str:
        return text
