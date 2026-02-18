import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import openai

load_dotenv()

def generate_text_for_ia_by_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = reader.pages
    text = ""
    for page in pages:
        text += page.extract_text()
    return text


if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(__file__), "thesis.pdf")
    text = generate_text_for_ia_by_pdf(pdf_path)
    
    # Salvar o texto em um arquivo .txt
    txt_path = os.path.splitext(pdf_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Texto extraído e salvo em: {txt_path}")
    print(f"Total de caracteres: {len(text)}")