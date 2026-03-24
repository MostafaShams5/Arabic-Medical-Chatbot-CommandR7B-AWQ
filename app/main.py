import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers.utils.import_utils

def is_torch_fx_available():
    """Deprecated: kept for backwards compatibility with trust_remote_code models."""
    return True

# Inject the mocked function back into the library before the model loads it
transformers.utils.import_utils.is_torch_fx_available = is_torch_fx_available

print("Loading BGE-M3 Embedding Model...")
# BGE-M3 handles both semantic (dense) and keyword (sparse) search
embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print("Connecting to Qdrant Database...")
# Points to the database folder that will be mounted inside our Docker container
DB_PATH = os.getenv("QDRANT_PATH", "./medical_qdrant_db")
client = QdrantClient(path=DB_PATH)
COLLECTION_NAME = "arabic_medical_HybridRAG"

# =========================================================
# 2. INITIALIZE NATIVE TRANSFORMERS LLM 
# =========================================================
print("Loading Model (tawkeed-egy-medical-4b) in native FP16...")
MODEL_ID = "Shams03/tawkeed-egy-medical-4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Loading in native float16. A 4B model takes ~8GB VRAM, fitting easily on a 16GB T4 GPU.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()
print("AI Engine Online and Ready.")

# =========================================================
# 3. FASTAPI SETUP & DATA CONTRACTS
# =========================================================
app = FastAPI(title="Tawkeed Medical RAG API", version="1.0")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    source: str
    page: str

system_preamble = """You are a medical assistant. You must follow these instructions. Do not deviate.

### RULE 1: Non-Medical Questions
If the user asks anything that is NOT about medicine, health, or biology, you MUST output ONLY:
"عذراً، أنا مخصص للإجابة على الاستفسارات الطبية فقط."

### RULE 2: Mandatory Citation Format
If you are provided with a source AND a page number in the retrieved context:
- You MUST start your response with this exact Arabic formula:
  "بناءً على [اسم المصدر]، صفحة [رقم الصفحة]: "
- Replace brackets with actual values from the context.
- If no page number is provided, omit "صفحة [رقم]" and write only: "بناءً على [اسم المصدر]: "

### RULE 3: Handling Missing or Irrelevant Sources
If NO source is provided, or the source is irrelevant to the question:
1. Provide a medically accurate answer based on your general knowledge.
2. End your answer with this exact Arabic phrase:
   "يجب عليك استشارة طبيب مختص للحصول على تشخيص دقيق."

### CRITICAL OUTPUT RULES (MOST IMPORTANT)
1. Think silently.
2. Output ONLY the final answer wrapped inside <final_answer> and </final_answer> tags.
"""

# =========================================================
# 4. THE MAIN ENDPOINT
# =========================================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.question
        
        # --- A. HYBRID RETRIEVAL ---
        embeddings = embedding_model.encode(query, return_dense=True, return_sparse=True)
        sparse_dict = embeddings['lexical_weights']
        
        # Qdrant requires indices to be integers and values to be floats
        sparse_vec = qdrant_models.SparseVector(
            indices=[int(k) for k in sparse_dict.keys()],
            values=[float(v) for v in sparse_dict.values()]
        )

        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qdrant_models.Prefetch(query=embeddings['dense_vecs'].tolist(), using="dense", limit=4),
                qdrant_models.Prefetch(query=sparse_vec, using="sparse", limit=4),
            ],
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=2
        ).points

        context_text, source_name, page_num = "", "", ""
        if search_results:
            pt = search_results[0]
            page_num = pt.payload.get('page', pt.payload.get('page_number', '1'))
            context_text = pt.payload.get('text', pt.payload.get('page_content', pt.payload.get('content', '')))
            source_name = pt.payload.get('source', pt.payload.get('file_name', pt.payload.get('document', 'غير معروف')))

        # --- B. PROMPT CONSTRUCTION ---
        user_content = f"السؤال: {query}\n\n"
        if context_text:
            user_content += f"السياق الطبي المسترجع:\n{context_text}\nالمصدر: {source_name}\nالصفحة: {page_num}\n"

        # Manually building ChatML to guarantee the format is respected perfectly
        prompt = f"<|im_start|>system\n{system_preamble}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

        # --- C. TRANSFORMERS GENERATION ---
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Slice to extract only the generated tokens (ignoring the input prompt)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        final_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean the output based on your specific tags
        clean_answer = final_output
        if "<final_answer>" in clean_answer:
            clean_answer = clean_answer.split("<final_answer>")[-1]
        if "</final_answer>" in clean_answer:
            clean_answer = clean_answer.split("</final_answer>")[0]
            
        clean_answer = clean_answer.strip()

        return ChatResponse(
            question=query,
            answer=clean_answer,
            source=str(source_name),
            page=str(page_num)
        )
        
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during AI processing.")

# =========================================================
# 5. SERVER EXECUTION
# =========================================================
if __name__ == "__main__":
    # Runs the server on port 8000, listening to all incoming IP addresses
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
