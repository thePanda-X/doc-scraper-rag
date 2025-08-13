import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def is_valid_url(url, base_url):
    """
    Check if the URL is valid and belongs to the same domain as the base URL.
    """
    parsed = urlparse(url)
    base_parsed = urlparse(base_url)
    return bool(parsed.netloc) and parsed.netloc == base_parsed.netloc

def extract_links(soup, base_url):
    """
    Extract all valid links from the BeautifulSoup object.
    """
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        if is_valid_url(full_url, base_url):
            links.add(full_url)
    return links

def extract_code_and_pre_from_url(url):
    """
    Extract code and pre snippets from a given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        code_elements = soup.find_all('code')
        pre_elements = soup.find_all('pre')

        code_snippets = [element.get_text() for element in code_elements]
        pre_snippets = [element.get_text() for element in pre_elements]

        full_snippets = code_snippets + pre_snippets
        full_snippets = [snippet for snippet in full_snippets if len(snippet.split()) > 1]

        return full_snippets
    except requests.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def crawl_site(base_url, max_pages=10):
    """
    Crawl the site starting from the base URL and extract code and pre snippets from each page.
    """
    visited = set()
    to_visit = {base_url}
    all_snippets = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        print(f"Crawling: {current_url}")
        visited.add(current_url)

        snippets = extract_code_and_pre_from_url(current_url)
        all_snippets.extend(snippets)

        try:
            response = requests.get(current_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links = extract_links(soup, base_url)
            to_visit.update(links - visited)
        except requests.RequestException as e:
            print(f"An error occurred while fetching the URL: {e}")

    return all_snippets

# Example usage
base_url = 'https://fastapi.tiangolo.com'
#base_url = 'https://python.langchain.com/docs'
snippets = crawl_site(base_url)

file_name = base_url.split(".")[-2]

with open(f'{file_name}_snippets.txt', 'w') as f:
    for snippet in snippets:
        f.write("=" * 20 + '\n')
        f.write(snippet + '\n')
    print(f"Saved snippets to {file_name}_snippets.txt")
