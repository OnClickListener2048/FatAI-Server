import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote

import httpx

from app.models import WebSearchResult
from app.services.errors import ServiceError

_TAG_PATTERN = re.compile(r"<[^>]+>")


class BingRssSearchService:
    """Keyless search provider backed by Bing's RSS output.

    HTML scraping providers (DuckDuckGo, Bing HTML, Baidu) are IP-blocked from many residential
    and datacenter networks. Bing's `format=rss` endpoint still returns server-rendered results
    without JavaScript or an API key. Production deployments can replace this service with an
    approved search provider while retaining the API contract exposed to clients and tools.
    """

    SEARCH_URL = "https://www.bing.com/search"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def search(self, query: str, max_results: int) -> list[WebSearchResult]:
        query = query.strip()
        if not query:
            raise ServiceError("INVALID_REQUEST", "query must not be blank.", 400)

        try:
            response = await self._client.get(
                self.SEARCH_URL,
                params={"q": query, "format": "rss"},
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ServiceError("SEARCH_UNAVAILABLE", "Search provider is unavailable.") from error

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as error:
            raise ServiceError("SEARCH_UNAVAILABLE", "Search provider returned an invalid feed.") from error

        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()
            if not title or not url or url in seen_urls:
                continue
            description = _TAG_PATTERN.sub(" ", item.findtext("description") or "").strip()
            seen_urls.add(url)
            results.append(
                WebSearchResult(
                    title=title,
                    snippet=re.sub(r"\s+", " ", description),
                    url=url,
                    source="bing",
                )
            )
            if len(results) == max_results:
                break
        return results


class WeatherService:
    def __init__(self, search: BingRssSearchService) -> None:
        self._search = search

    async def weather(self, location: str, max_results: int) -> list[WebSearchResult]:
        results = await self._search.search(f"{location.strip()} weather timeanddate", max_results=10)
        preferred = sorted(results, key=lambda result: "timeanddate.com" not in result.url)
        return [
            result.model_copy(update={"source": "timeanddate"})
            if "timeanddate.com" in result.url
            else result
            for result in preferred[:max_results]
        ]
