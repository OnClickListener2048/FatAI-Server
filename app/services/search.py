from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from app.models import WebSearchResult
from app.services.errors import ServiceError


class DuckDuckGoSearchService:
    """Development-only keyless search provider.

    Production deployments should replace this service with an approved search provider while
    retaining the API contract exposed to clients and LangChain tools.
    """

    SEARCH_URL = "https://html.duckduckgo.com/html/"
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
                params={"q": query},
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ServiceError("SEARCH_UNAVAILABLE", "Search provider is unavailable.") from error

        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        document = BeautifulSoup(response.text, "html.parser")
        for anchor in document.select("a.result__a"):
            url = self._destination_url(anchor.get("href", ""))
            title = anchor.get_text(" ", strip=True)
            if not url or not title or url in seen_urls:
                continue
            container = anchor.find_parent(class_="result")
            snippet_node = container.select_one("a.result__snippet") if container else None
            results.append(
                WebSearchResult(
                    title=title,
                    snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
                    url=url,
                    source="duckduckgo",
                )
            )
            seen_urls.add(url)
            if len(results) == max_results:
                break
        return results

    @staticmethod
    def _destination_url(raw_url: str) -> str | None:
        if raw_url.startswith("//"):
            raw_url = f"https:{raw_url}"
        parsed = urlparse(raw_url)
        if parsed.path.startswith("/l/"):
            destination = parse_qs(parsed.query).get("uddg", [""])[0]
            raw_url = unquote(destination)
        return raw_url if raw_url.startswith(("https://", "http://")) else None


class WeatherService:
    def __init__(self, search: DuckDuckGoSearchService) -> None:
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
