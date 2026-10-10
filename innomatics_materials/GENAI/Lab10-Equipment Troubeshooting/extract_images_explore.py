import fitz #PyMuPDf
import easyocr
import numpy as np
reader = easyocr.Reader(['en'])


pdf_path = "Equipment Manuals/2200 Series Conveyors Installation Maintenance & Parts Manual.pdf"

#Extract the information from PDF
pdf = fitz.open(str(pdf_path))
each_page = 5
page_number = each_page +1
page = pdf[each_page]


# print(page.cluster_drawings())

rects = page.cluster_drawings()
MIN_FIGURE_SIZE_PT = 40
candidate_rects = [r for r in page.cluster_drawings() if r.width > MIN_FIGURE_SIZE_PT and r.height > MIN_FIGURE_SIZE_PT]

for fig_num,rect  in enumerate(candidate_rects,1):
    pix = page.get_pixmap(dpi=150, clip=rect)  #DPI -- Dots per inch
    combined_text = ""
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
    pix.height, pix.width, pix.n)
    result = reader.readtext(img_array, paragraph=True)
    for each_res in result:
        coord,text = each_res
        combined_text += text
    if len(combined_text)>20:
        continue
    image_path = "images/"+ f"page_{page_number:04d}_fig_{fig_num}.png"
    pix.save(str(image_path))