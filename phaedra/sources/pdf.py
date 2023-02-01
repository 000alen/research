import pdfplumber

from phaedra.sources import SourceInterface


class PDF(SourceInterface):
    @staticmethod
    def extract_text(path: str) -> str:
        with pdfplumber.open(path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages])
