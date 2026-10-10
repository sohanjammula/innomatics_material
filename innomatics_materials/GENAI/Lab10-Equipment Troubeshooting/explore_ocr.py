# import easyocr

# reader = easyocr.Reader(['en'])
# result = reader.readtext('/home/abhishek/GenAI_AgenticAI_Labs_Z2G/Lab10-Equipment Troubeshooting/images/page_0006_fig_3.png', paragraph=True)

# print(result)
# for each_res in result:
#     coord,text = each_res
#     print(text)


import fitz  # PyMuPDF
import easyocr

PDF_PATH = "Equipment Manuals/Conveyors and Related Equipment (scanned).pdf"
PAGE_INDEX = 5  # first page of this file


def main():
    pdf = fitz.open(PDF_PATH)
    page = pdf[PAGE_INDEX]

    print(f"pdfminer/pypdf-style text extraction on page {PAGE_INDEX + 1}:")
    print("Fitz text:",repr(page.get_text()))  # empty - there's no text layer to extract

    image_path = "demo_ocr_page.png"
    page.get_pixmap(dpi=150).save(image_path)
    pdf.close()

    print("\nLoading EasyOCR model (first run downloads it)...")
    reader = easyocr.Reader(["en"])
    lines = reader.readtext(image_path, detail=0, paragraph=True)

    print(f"\nEasyOCR recovered {len(lines)} lines of text from the page image:\n")
    for line in lines:
        print(" -", line)


if __name__ == "__main__":
    main()
