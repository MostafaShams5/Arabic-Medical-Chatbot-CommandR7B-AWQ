import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def download_pdfs_from_medlineplus(url, output_folder="medline_pdfs"):
    """
    Downloads all PDF files linked on a specific MedlinePlus page.
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    print(f"Fetching page: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return


    soup = BeautifulSoup(response.text, 'html.parser')
    
    links = soup.find_all('a', href=True)
    
    pdf_links = []
    for link in links:
        href = link['href']

        if href.lower().strip().endswith('.pdf'):
            full_url = urljoin(url, href)
            pdf_links.append(full_url)

    pdf_links = list(set(pdf_links))
    
    print(f"Found {len(pdf_links)} PDF links. Starting download...")


    for i, pdf_url in enumerate(pdf_links, 1):
        try:
            # Extract a clean filename from the URL
            parsed_url = urlparse(pdf_url)
            filename = os.path.basename(parsed_url.path)
            
            if not filename or filename.strip() == "":
                filename = f"document_{i}.pdf"
            
            save_path = os.path.join(output_folder, filename)

            if os.path.exists(save_path):
                print(f"[{i}/{len(pdf_links)}] Skipped (already exists): {filename}")
                continue

            print(f"[{i}/{len(pdf_links)}] Downloading: {filename}")
            
            with requests.get(pdf_url, headers=headers, stream=True) as pdf_response:
                pdf_response.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Politeness: Sleep briefly to avoid overwhelming the server
            time.sleep(1) 

        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")


if __name__ == "__main__":
    target_url = "https://medlineplus.gov/languages/arabic.html"
    download_pdfs_from_medlineplus(target_url)
