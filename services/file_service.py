import PyPDF2
import io

class FileService:
    async def extract_text(self, content: bytes, filename: str) -> str:
        if filename.lower().endswith('.pdf'):
            return await self._extract_from_pdf(content)
        elif filename.lower().endswith('.txt'):
            return content.decode('utf-8')
        else:
            raise ValueError("Unsupported file format")
    
    async def _extract_from_pdf(self, content: bytes) -> str:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
