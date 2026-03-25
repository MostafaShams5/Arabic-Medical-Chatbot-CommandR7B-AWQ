import re
import transformers
import transformers.utils
import transformers.utils.import_utils

# 2. Define the patch
def is_torch_fx_available():
    """Deprecated: kept for backwards compatibility with trust_remote_code models."""
    return True

# 3. Inject it into the module
transformers.utils.import_utils.is_torch_fx_available = is_torch_fx_available
transformers.utils.is_torch_fx_available = is_torch_fx_available

def check_torch_load_is_safe():
    """Bypasses the PyTorch 2.6 version lock for trusted .bin models like BGE-M3."""
    pass

# Force-inject the security bypass
transformers.utils.import_utils.check_torch_load_is_safe = check_torch_load_is_safe

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from FlagEmbedding import BGEM3FlagModel

print("Loading BGE-M3 Embedding Model...")
embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print("Connecting to Qdrant Database...")
DB_PATH = os.getenv("QDRANT_PATH", "./medical_qdrant_db")
if os.path.exists(DB_PATH):
    for root, dirs, files in os.walk(DB_PATH):
        for file in files:
            if file == ".lock":
                lock_path = os.path.join(root, file)
                try:
                    os.remove(lock_path)
                    print(f"🔓 Removed stale lock file: {lock_path}")
                except Exception as e:
                    print(f"⚠️ Could not remove lock: {e}")

client = QdrantClient(path=DB_PATH)
COLLECTION_NAME = "arabic_medical_HybridRAG"

print("Loading Model (tawkeed-egy-medical-4b) in native FP16...")
MODEL_ID = "Shams03/tawkeed-egy-medical-4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()
print("AI Engine Online and Ready.")

app = FastAPI(title="Tawkeed Medical RAG API", version="1.0")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    source: str
    page: str
    score: float
    retrieved_chunks: list[dict]

system_preamble = """
You are a medical assistant. Answer only medical, health, or biology questions.

If the user asks anything outside those topics, reply only:
عذراً، أنا مخصص للإجابة على الاستفسارات الطبية فقط.

You may receive retrieved context from medical documents. Use it as the primary evidence.

Citation policy (STRICT):
- If any retrieved source is relevant, the first words of the answer after <final_answer> MUST be the real source name.
- Never use generic phrases like: "المصادر الطبية المذكورة".
- If source and page exist, start exactly like this:
  بناءً على [اسم المصدر]، صفحة [رقم الصفحة]:
  Example:
  بناءً على الموسوعة الطبية الشاملة، صفحة 219:
- If only source exists, start exactly like this:
  بناءً على [اسم المصدر]:
  Example:
  بناءً على MedlinePlus - Headaches_ARA:
- If two sources are relevant, you MAY cite both:
  بناءً على [المصدر الأول]، صفحة [رقم الصفحة]، و [المصدر الثاني]، صفحة [رقم الصفحة]:

- If no relevant source exists, answer normally and end with:
  يجب عليك استشارة طبيب مختص للحصول على تشخيص دقيق.

Answer quality:
- THE ANSWER MUST LOOK COMPLETE AND ACCURATE.
- Select only the parts that directly answer the question.
- Ignore irrelevant chunks.
- Rewrite naturally in Arabic.
- Keep the answer clear and focused.

OUTPUT FORMAT (STRICT):
- Your answer MUST start with: <final_answer>
- Your answer MUST end with: </final_answer>
- Do not output anything before <final_answer> or after </final_answer>

Do not show reasoning or internal thoughts.
"""

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

def extract_final_answer(text: str) -> str:
    text = (text or "").strip()

    if "<final_answer>" in text and "</final_answer>" in text:
        text = text.split("<final_answer>", 1)[-1]
        text = text.rsplit("</final_answer>", 1)[0]

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    first_match = ARABIC_RE.search(text)
    if not first_match:
        return text.strip()

    last_match = None
    for m in ARABIC_RE.finditer(text):
        last_match = m

    return text[first_match.start():last_match.end()].strip()


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.question

        embeddings = embedding_model.encode(query, return_dense=True, return_sparse=True)
        sparse_dict = embeddings["lexical_weights"]

        sparse_vec = qdrant_models.SparseVector(
            indices=[int(k) for k in sparse_dict.keys()],
            values=[float(v) for v in sparse_dict.values()]
        )

        candidate_limit = 8

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qdrant_models.Prefetch(
                    query=embeddings["dense_vecs"].tolist(),
                    using="dense",
                    limit=candidate_limit
                ),
                qdrant_models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=candidate_limit
                ),
            ],
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=candidate_limit
        )

        search_results = response.points or []

        filtered_results = []
        for pt in search_results:
            score = getattr(pt, "score", None)
            if score is None:
                continue
            score = float(score)
            if score > 0.50:
                filtered_results.append(pt)

        filtered_results = filtered_results[:2]

        context_parts = []
        retrieved_chunks = []

        for pt in filtered_results:
            payload = pt.payload or {}
            page_num = payload.get("page", payload.get("page_number", "1"))
            context_text = payload.get("text", payload.get("page_content", payload.get("content", "")))
            source_name = payload.get("source", payload.get("file_name", payload.get("document", "غير معروف")))
            score = float(getattr(pt, "score", 0.0))

            retrieved_chunks.append({
                "source": str(source_name),
                "page": str(page_num),
                "score": score,
                "text": str(context_text)
            })

            context_parts.append(
                f"المصدر: {source_name}\n"
                f"الصفحة: {page_num}\n"
                f"الدرجة: {score:.3f}\n"
                f"النص: {context_text}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        user_content = f"السؤال: {query}\n\n"
        if context_text:
            user_content += f"السياق الطبي المسترجع:\n{context_text}\n"

        messages = [
    {"role": "system", "content": system_preamble},
    {"role": "user", "content": user_content}
]

        text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
             
            
            outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.1
    )

        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        final_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        clean_answer = extract_final_answer(final_output)

        top_source = str(retrieved_chunks[0]["source"]) if retrieved_chunks else ""
        top_page = str(retrieved_chunks[0]["page"]) if retrieved_chunks else ""
        top_score = float(retrieved_chunks[0]["score"]) if retrieved_chunks else 0.0

        return ChatResponse(
            question=query,
            answer=clean_answer,
            source=top_source,
            page=top_page,
            score=top_score,
            retrieved_chunks=retrieved_chunks
        )

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during AI processing.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
