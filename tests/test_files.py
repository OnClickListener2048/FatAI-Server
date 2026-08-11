import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from tests import _test_db

os.environ["DATABASE_URL"] = _test_db.DATABASE_URL
os.environ["UPLOAD_DIRECTORY"] = _test_db.UPLOAD_DIRECTORY

from fastapi.testclient import TestClient

from app.main import create_app


class FakeDocumentService:
    async def read(self, display_name: str, mime_type: str, content: bytes):
        return AsyncMock()  # replaced below; kept for signature parity


class FileUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())
        self.client.__enter__()
        response = self.client.post(
            "/v1/auth/device",
            json={"device_id": "123e4567-e89b-12d3-a456-426614174099", "display_name": "Files test device"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.headers = {"Authorization": f"Bearer {response.json()['access_token']}"}

        async def fake_read(self, display_name: str, mime_type: str, content: bytes):
            return {"displayName": display_name, "markdown": f"# {display_name}\nconverted {len(content)} bytes"}

        fake_service = type("Fake", (), {"read": fake_read})()
        self.client.app.state.document_service = fake_service

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def test_upload_then_read_by_file_id(self) -> None:
        # 上传(模拟 S3 存储) -> 返回 file_id
        upload = self.client.post(
            "/v1/files",
            headers=self.headers,
            files={"file": ("diploma.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        asset = upload.json()
        file_id = asset["id"]
        self.assertEqual(asset["display_name"], "diploma.pdf")
        self.assertEqual(asset["mime_type"], "application/pdf")
        self.assertTrue(Path(asset["storage_path"]).is_file())

        # 按 file_id 读取转换(服务端自己读存储, 客户端不暴露本地路径)
        read = self.client.post(f"/v1/files/{file_id}/read", headers=self.headers)
        self.assertEqual(read.status_code, 200, read.text)
        self.assertEqual(read.json()["displayName"], "diploma.pdf")
        self.assertIn("converted 19 bytes", read.json()["markdown"])

    def test_read_requires_ownership(self) -> None:
        upload = self.client.post(
            "/v1/files",
            headers=self.headers,
            files={"file": ("private.txt", b"secret", "text/plain")},
        )
        file_id = upload.json()["id"]

        # 无鉴权
        anon = self.client.post(f"/v1/files/{file_id}/read")
        self.assertEqual(anon.status_code, 401, anon.text)

        # 他人资产 404
        other = self.client.post(
            "/v1/auth/device",
            json={"device_id": "123e4567-e89b-12d3-a456-426614174098", "display_name": "Other device"},
        )
        other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
        denied = self.client.post(f"/v1/files/{file_id}/read", headers=other_headers)
        self.assertEqual(denied.status_code, 404, denied.text)
