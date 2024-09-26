import PyPDF2

class FileParser:
    def parse_pdf(self, file):
        """Parse text from a PDF file."""
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

    def parse_txt(self, file):
        """Parse text from a TXT file."""
        return file.read().decode("utf-8")

    def parse_file(self, uploaded_file):
        """Parse uploaded file based on its type."""
        if uploaded_file.type == "application/pdf":
            return self.parse_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            return self.parse_txt(uploaded_file)
        else:
            return None
