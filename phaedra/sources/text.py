from phaedra.sources import SourceInterface


class Text(SourceInterface):
    @staticmethod
    def extract_text(text: str) -> str:
        return text
