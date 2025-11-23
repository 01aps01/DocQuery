from transformers import BlipProcessor, BlipForConditionalGeneration

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import fitz
import uuid

import pytesseract
from PIL import Image

app = FastAPI()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is working 🚀"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    doc = fitz.open(pdf_path)
    extracted_text = ""
    image_paths = []

    for page_num, page in enumerate(doc):
        extracted_text += page.get_text()

        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            image_filename = f"image_{page_num}_{img_index}_{uuid.uuid4().hex}.png"

            if pix.n < 5:
                pix.save(image_filename)
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(image_filename)

            image_paths.append(image_filename)

    ocr_text = ""
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            ocr_text += text + "\n"
        except Exception as e:
            print("OCR error:", e)

    captions = []
    for image_path in image_paths:
        try:
            raw_image = Image.open(image_path).convert("RGB")
            inputs = processor(images=raw_image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        except Exception as e:
            print("Captioning error:", e)


    return {
    "filename": file.filename,
    "text_preview": extracted_text[:300],
    "total_images_found": len(image_paths),
    "ocr_preview": ocr_text[:300],
    "image_captions": captions[:3] 
}


