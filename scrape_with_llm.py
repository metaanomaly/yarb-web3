import json
import os
import httpx
import logging
import re
from pathlib import Path
from trafilatura import fetch_url, extract  # Clean HTML/text
import openai  # Or llama.cpp/ollama for open-source models
from dotenv import load_dotenv
import colorlog
from datetime import datetime

# Configure logging with colors
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
))

# Create logger
logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY", None)

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

def get_latest_archive_file():
    """Get the most recently created markdown file from archive directory"""
    root_path = Path(__file__).absolute().parent
    archive_path = root_path.joinpath('archive')
    
    latest_file = None
    latest_time = 0
    
    # Walk through all year directories
    for year_dir in archive_path.glob('*'):
        if not year_dir.is_dir():
            continue
            
        # Check all md files in year directory
        for md_file in year_dir.glob('*.md'):
            file_time = md_file.stat().st_mtime
            if file_time > latest_time:
                latest_time = file_time
                latest_file = md_file
                
    return latest_file

def extract_urls_from_md(md_file):
    """Extract all URLs and their titles from markdown file"""
    if not md_file:
        logger.error("No markdown file provided")
        return []
        
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find all markdown links [title](url)
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(pattern, content)
    
    # Create list of tuples (url, title)
    url_data = [(url, title) for title, url in matches]
    logger.info(f"Found {len(url_data)} URLs in {md_file.name}")
    return url_data

def extract_with_llm(html: str, url: str) -> dict:
    """Use LLM to extract structured data from HTML."""
    logger.info(f"Processing URL: {url}")
    
    # Step 1: Extract clean text
    logger.debug("Extracting clean text from HTML")
    text = extract(html, include_links=True, include_tables=True)
    if not text:
        logger.warning("Failed to extract text, falling back to raw HTML")
        text = html

    # Step 2: Prepare LLM prompt
    logger.debug("Preparing LLM prompt")
    prompt = f"""
    You are a crypto news extraction assistant. Analyze the following content from {url} and return:
    - Title
    - Detailed summary extracting valuable information, data, and insights from the content
    - Key entities (e.g., projects, people, tokens)
    - Sentiment (bullish/bearish/neutral)
    - Category (e.g., DeFi, NFTs, Regulation)
    Format the output as JSON. Do not include markdown.

    Content:
    {text}
    """

    # Step 3: Call LLM
    logger.info("Calling OpenAI API")
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a crypto news analyst."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        logger.debug("Successfully received OpenAI response")
        
        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        logger.info("Successfully extracted structured data")
        return result
    except Exception as e:
        logger.error(f"Error processing with LLM: {str(e)}")
        return {"error": str(e)}

async def scrape_url(url: str, title: str = None):
    """Fetch and process a single URL."""
    logger.info(f"Starting to process URL: {url}")
    try:
        if "t.me/" in url:
            # For Telegram links, use the title as content directly
            if not title:
                logger.error(f"No title provided for Telegram link: {url}")
                return None
            
            logger.info(f"Processing Telegram message: {title}")
            result = extract_with_llm(title, url)  # Pass title as content
            result["url"] = url
            logger.info(f"Successfully processed Telegram message")
            return result
        else:
            # Regular web URLs - fetch and process HTML
            html = fetch_url(url)
            if not html:
                logger.error(f"Failed to fetch HTML from {url}")
                return None

            result = extract_with_llm(html, url)
            result["url"] = url
            logger.info(f"Successfully processed {url}")
            return result
            
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        return {"url": url, "error": str(e)}

async def main():
    logger.info("Starting web scraping process")
    
    # Get latest archive file and extract URLs with titles
    latest_md = get_latest_archive_file()
    if not latest_md:
        logger.error("No archive files found")
        return
        
    logger.info(f"Processing latest archive: {latest_md}")
    url_data = extract_urls_from_md(latest_md)
    
    if not url_data:
        logger.error("No URLs found in archive file")
        return
        
    logger.info(f"Processing {len(url_data)} URLs")

    # Process URLs
    async with httpx.AsyncClient() as client:
        tasks = [scrape_url(url, title) for url, title in url_data]
        results = await asyncio.gather(*tasks)

    # Save results
    successful_results = [r for r in results if r]
    logger.info(f"Successfully processed {len(successful_results)} out of {len(url_data)} URLs")
    
    output_file = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-llm_processed_articles.json"
    with open(output_file, "w") as f:
        json.dump(successful_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())