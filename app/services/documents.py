import httpx

from app.models import DocumentReadResponse
from app.services.errors import ServiceError


class DoclingDocumentService:
    def __init__(self, client: httpx.AsyncClient, server_url: str, max_size_bytes: int) -> None:
        self._client = client
        self._server_url = server_url.rstrip("/")
        self._max_size_bytes = max_size_bytes

    async def read(self, display_name: str, mime_type: str, content: bytes) -> DocumentReadResponse:
        if not display_name.strip():
            raise ServiceError("INVALID_REQUEST", "A filename is required.", 400)
        if not content:
            raise ServiceError("INVALID_REQUEST", "The uploaded file is empty.", 400)
        if len(content) > self._max_size_bytes:
            raise ServiceError("INVALID_REQUEST", "The uploaded file exceeds the 50 MB limit.", 413)

        try:
            response = await self._client.post(
                f"{self._server_url}/v1/convert/file",
                files={"files": (display_name, content, mime_type or "application/octet-stream")},
                data={"to_formats": "md", "image_export_mode": "placeholder", "do_ocr": "true"},
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ServiceError("DOCLING_UNAVAILABLE", "Docling service is unavailable.") from error

        try:
            payload = response.json()
            markdown = payload["document"]["md_content"].strip()
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            message = self._extract_docling_error(response)
            raise ServiceError("DOCLING_FAILED", message) from error
        if not markdown:
            raise ServiceError("DOCLING_FAILED", "Docling did not return Markdown for the document.")
        return DocumentReadResponse(displayName=display_name, markdown=markdown)

    @staticmethod
    def _extract_docling_error(response: httpx.Response) -> str:
        try:
            errors = response.json().get("errors", [])
            if errors and errors[0].get("error_message"):
                return str(errors[0]["error_message"])
        except (ValueError, AttributeError, IndexError):
            pass
        return "Docling returned an invalid conversion response."
