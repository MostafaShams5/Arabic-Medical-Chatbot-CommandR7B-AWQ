import os
import json
import re
import uuid
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract


OUTPUT_FILE  = "MedicineBooksChunks.jsonl"
DPI          = 300         
MAX_WORKERS  = 4            
WRITE_BUFFER = 30          


target_files = [
    f for f in os.listdir(".")
    if os.path.isfile(f) and f != OUTPUT_FILE and not f.endswith(".py")
]


def process_page(file_path: str, book_title: str, page_index: int) -> list[dict]:
    """Convert one page → OCR → chunk. Returns a list of record dicts."""
    records = []
    try:
        pages = convert_from_path(
            file_path, dpi=DPI,
            first_page=page_index, last_page=page_index
        )
        if not pages:
            return records

        page_image = pages[0]
        width, height = page_image.size
        crop_box      = (0, height * 0.1, width, height * 0.9)
        cropped_image = page_image.crop(crop_box)

        # Free the full-page image immediately after cropping
        del pages, page_image
        gc.collect()

        raw_text         = pytesseract.image_to_string(cropped_image, lang="ara+eng")
        del cropped_image
        gc.collect()


        text_no_newlines = raw_text.replace("\n", " ")
        cleaned_text     = re.sub(
            r"[^\u0600-\u06FFa-zA-Z0-9\s\.,:;\-\(\)\?!]", "", text_no_newlines
        )
        final_text = re.sub(r"\s+", " ", cleaned_text).strip()

        words = final_text.split()
        if not words:
            return records


        page_chunks: list[list[str]] = []
        while len(words) > 200:
            if len(words) - 200 < 120:
                half = len(words) // 2
                page_chunks.append(words[:half])
                words = words[half:]
            else:
                page_chunks.append(words[:200])
                words = words[200:]
        if words:
            page_chunks.append(words)

        for chunk_words in page_chunks:
            word_count = len(chunk_words)
            if 120 <= word_count <= 200:
                records.append({
                    "chunk_id":   uuid.uuid4().hex,
                    "book_name":  book_title,
                    "page_number": page_index,
                    "word_count": word_count,
                    "content":    " ".join(chunk_words),
                })

    except Exception:
        pass

    return records



with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:

    write_buffer: list[str] = []

    def flush_buffer():
        if write_buffer:
            out_f.write("\n".join(write_buffer) + "\n")
            out_f.flush()
            write_buffer.clear()

    for file_path in target_files:
        book_title = os.path.basename(file_path)

        try:
            info        = pdfinfo_from_path(file_path)
            total_pages = int(info["Pages"])
            page_range  = range(11, total_pages + 1)

            completed = 0
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_page, file_path, book_title, p): p
                    for p in page_range
                }

                for future in as_completed(futures):
                    records = future.result()
                    for rec in records:
                        write_buffer.append(json.dumps(rec, ensure_ascii=False))

                    completed += 1
                    if len(write_buffer) >= WRITE_BUFFER:
                        flush_buffer()

                    if completed % 50 == 0:
                        print(f"  … {completed}/{len(page_range)} pages done")

            flush_buffer()  


        except Exception as e:
            print(f" Skipped ({e})")
            flush_buffer()


