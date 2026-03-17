import json
import uuid
import re
from collections import defaultdict

def clean_text(text):
    """
    Cleans the text by removing weird encoding artifacts, hidden control characters,
    and compressing multiple spaces/newlines into a single space.
    """
    if not text:
        return ""
    

    cleaned = re.sub(r'[^\w\s،؛؟\.\,\!\?\(\)\-\:\/]', ' ', text)
    
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def process_and_chunk(input_file, output_file):
    source_groups = defaultdict(list)
    

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
        
            item['content'] = clean_text(item.get('content', ''))
            source_groups[item['source']].append(item)

    new_chunks = []
    max_words_found = 0

    for source, items in source_groups.items():
        is_null_source = all(item.get('page_number') is None for item in items)

        if is_null_source:
            all_words = []
            for item in items:
                all_words.extend(item['content'].split())

            chunk_size = 220
            overlap = 50
            step = chunk_size - overlap 

            for i in range(0, len(all_words), step):
                chunk_words = all_words[i:i + chunk_size]
                if not chunk_words:
                    break

                word_count = len(chunk_words)
                
                if word_count > max_words_found:
                    max_words_found = word_count

                new_chunks.append({
                    "chunk_id": uuid.uuid4().hex,
                    "source": source,
                    "page_number": None,
                    "word_count": word_count,
                    "content": " ".join(chunk_words)
                })
                
                if i + chunk_size >= len(all_words):
                    break

        else:
            try:
                items = sorted(items, key=lambda x: int(x['page_number']) if x['page_number'] is not None else 0)
            except ValueError:
                pass 

            for i in range(len(items)):
                current_item = items[i]
                current_words = current_item['content'].split()
                original_page = current_item['page_number']

                overlap_words = []
                next_page = None
                words_needed = 50
                
                for j in range(i + 1, len(items)):
                    next_words = items[j]['content'].split()
                    take_words = next_words[:words_needed]
                    overlap_words.extend(take_words)
                    
                    if take_words:
                        next_page = items[j]['page_number']
                        
                    words_needed -= len(take_words)
                    if words_needed <= 0:
                        break

                combined_words = current_words + overlap_words
                word_count = len(combined_words)
                
                if word_count > max_words_found:
                    max_words_found = word_count

                page_counts = {original_page: len(current_words)}
                if overlap_words and next_page is not None:
                    page_counts[next_page] = page_counts.get(next_page, 0) + len(overlap_words)

                majority_page = max(page_counts, key=page_counts.get) if page_counts else original_page

                new_chunks.append({
                    "chunk_id": uuid.uuid4().hex,
                    "source": source,
                    "page_number": majority_page,
                    "word_count": word_count,
                    "content": " ".join(combined_words)
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in new_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print(f"Processing complete. Data saved to {output_file}")

process_and_chunk('RAG.jsonl', 'RAG_cleaned_overlapped.jsonl')
