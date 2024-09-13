from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode
import os

app = Flask(__name__)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
list_of_true_sites = ["bbc.com", "reuters.com"]

# Функція для сканування сайту та отримання тексту
def scan_website(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com",
            "Connection": "keep-alive"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    except Exception as e:
        print(f"Error scanning website {url}: {e}")
        return ""

# Функція для пошуку результатів через Google
def search_google(query, site, start=0):
    query_with_site = f"{query} site:{site}"
    params = {
        'q': query_with_site,
        'start': start
    }
    url = f"https://www.google.com/search?{urlencode(params)}"
    
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for item in soup.find_all('a', href=True):
            href = item['href']
            if '/url?q=' in href:
                link = href.split('/url?q=')[1].split('&')[0]
                domain = urlparse(link).netloc.replace('www.', '')
                if any(true_site in domain for true_site in list_of_true_sites):
                    results.append(link)
                if len(results) >= 2:
                    break
        
        return results
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []

def run_model(input_string):
    input_ids = tokenizer(input_string, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])

def verify_claim(claim, summary):
    question = f"Is the following claim true: '{claim}' based on this text: '{summary}'?"
    return run_model(question)

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    result = {"query": query, "evidence": []}
    
    # Шукаємо докази по кожному сайту
    for site in list_of_true_sites:
        urls = search_google(query, site)
        
        if not urls:
            continue
        
        for url in urls:
            print(f"Scanning URL: {url}")
            text = scan_website(url)
            
            if len(text.split()) > 50:  # Мінімальна кількість слів для узагальнення
                try:
                    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                    verdict = verify_claim(query, summary)
                    
                    # Додаємо доказ до результату
                    result["evidence"].append({
                        "site": site,
                        "url": url,
                        "summary": summary,
                        "verdict": verdict
                    })
                except Exception as e:
                    print(f"Error summarizing text: {e}")
    
    if not result["evidence"]:
        return jsonify({"error": "No relevant evidence found"}), 404
    
    return jsonify(result)

if __name__ == "__main__":
    #app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
