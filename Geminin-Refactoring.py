import pandas as pd
import google.generativeai as genai
import time
import json
import os

API_KEY = "**********************"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
# model = genai.GenerativeModel('gemini-2.5-flash')

EXCEL_PATH = '/home/shams/Documents/AHD.xlsx'
OUTPUT_JSONL_PATH = '/home/shams/Documents/cleaned_medical_dataset.jsonl'

def clean_batch(batch_df):
    items_to_fix = []
    for idx, row in batch_df.iterrows():
        items_to_fix.append({
            "id": int(idx),
            "Question": str(row['Question']),
            "Answer": str(row['Answer'])
        })
    prompt = f"""
    أنت طبيب استشاري محترف
    
    2. ممنوع الإجابات القصيرة (مثل "أيوة عادي مفيش مشاكل"). يجب أن تشرح للمريض "السبب الطبي" بطريقة مبسطة، وتضيف نصيحة أو خطوة قادمة وذكر أي خطورة محتملة 
    3. يجب إذا كانت الإجابة الأصلية قصيرة أو سيئة، قم بتوسيعها علمياً بأسلوب طبي وافي.
    4. يمنع منعاً باتاً استخدام علامة الخط المائل العكسي (\\) في النص.
    
    يجب أن ترد بصيغة JSON فقط عبارة عن قائمة (List) تحتوي على كائنات. كل كائن يجب أن يضم:
    - "id": نفس المعرف بالظبط.
    - "Question": السؤال الأصلي كما هو بتعديل بسيط اذا كان يستدعي ذالك.
    - "Answer": إجابة طبية وافية، مفصلة، وصريحة.
    
    البيانات:
    {json.dumps(items_to_fix, ensure_ascii=False)}
    """
    
    # Auto-Retry Logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
            
        except Exception as e:
            error_message = str(e)

            if "429" in error_message or "quota" in error_message.lower():
                print(f" Rate limit hit. Waiting 60 seconds before trying again... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(60)
            else:
                print(f"Error: {e}")
                return None
                
    print("Failed after 3 retries. Skipping this batch.")
    return None

def main():
    print("Loading data...")
    df = pd.read_excel(EXCEL_PATH, usecols=["Question", "Answer"])
    
    processed_ids = set()
    if os.path.exists(OUTPUT_JSONL_PATH):
        with open(OUTPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line).get("id"))
                except:
                    pass


    with open(OUTPUT_JSONL_PATH, 'a', encoding='utf-8') as outfile:
        
        for i in range(0, len(df), 120):
            batch = df.iloc[i:i+120]
            batch_to_process = batch[~batch.index.isin(processed_ids)]
            
            if batch_to_process.empty:
                continue
                
            print(f"Processing rows {i} to {i+120}...")
            results = clean_batch(batch_to_process)
            
            if results:
                for item in results:
                    # Writes {"id": 1, "Question": "...", "Answer": "..."} straight to the file
                    outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                    outfile.flush()
            

            time.sleep(6)

if __name__ == "__main__":
    main()
