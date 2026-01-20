import asyncio
import requests
import bs4 as BeautifulSoup

req = requests.get('https://ease-int.com/')
html = req.text

content = BeautifulSoup(html, "html.parser")


async def root():
  return content.text

async def main():
  await root()

if __name__ == "__main__":
  asyncio.run(main())
