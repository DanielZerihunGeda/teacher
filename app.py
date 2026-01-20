import asyncio
import aiohttp
from bs4 import BeautifulSoup

async def fetch_text(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()

async def main():
    text = await fetch_text("https://ease-int.com/")
    print(text)

if __name__ == "__main__":
    asyncio.run(main())
