import json
import os
import httpx
import logging
from trafilatura import fetch_url, extract  # Clean HTML/text
import openai  # Or llama.cpp/ollama for open-source models
from dotenv import load_dotenv
import colorlog

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

async def scrape_url(url: str):
    """Fetch and process a single URL."""
    logger.info(f"Starting to scrape URL: {url}")
    try:
        # Fetch HTML
        html = fetch_url(url)
        if not html:
            logger.error(f"Failed to fetch HTML from {url}")
            return None

        # Extract with LLM
        result = extract_with_llm(html, url)
        result["url"] = url
        logger.info(f"Successfully processed {url}")
        return result
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        return {"url": url, "error": str(e)}

async def main():
    logger.info("Starting web scraping process")
    
    # Fetch URLs
    urls = [
        "https://thedefiant.io/news/markets/bitcoin-wallets-holding-100-to-1-000-btc-reach-record-high",
    ]
    logger.info(f"Processing {len(urls)} URLs")

    # Process URLs
    async with httpx.AsyncClient() as client:
        tasks = [scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks)

    # Save results
    successful_results = [r for r in results if r]
    logger.info(f"Successfully processed {len(successful_results)} out of {len(urls)} URLs")
    
    output_file = "llm_processed_articles.json"
    with open(output_file, "w") as f:
        json.dump(successful_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())