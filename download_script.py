import requests
import os
from datetime import datetime, timedelta

def download_file(url, output_dir):
    try:
        filename = url.split('/')[-1]
        output_path = os.path.join(output_dir, filename)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Successfully downloaded: {filename}")
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def generate_lichess_urls(months_back=2):
    months_back+=1
    base_url = "https://database.lichess.org/standard/lichess_db_standard_rated_{}-{:02d}.pgn.zst"
    urls = []
    
    current_date = datetime.now()
    for i in range(months_back):
        date = current_date - timedelta(days=i*30)
        url = base_url.format(date.year, date.month)
        urls.append(url)
    
    return urls[1:]

def main():
    output_dir = "pgn_files"
    os.makedirs(output_dir, exist_ok=True)
    
    urls = generate_lichess_urls()
    for url in urls:
        download_file(url, output_dir)

if __name__ == "__main__":
    main()