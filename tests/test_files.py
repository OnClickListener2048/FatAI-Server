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

    def test_upload_response_contains_absolute_url(self) -> None:
        # 服务端用请求的实际 base_url 生成附件 URL, 供其他设备经同步流取回文件
        upload = self.client.post(
            "/v1/files",
            headers=self.headers,
            files={"file": ("photo.png", b"\x89PNG fake", "image/png")},
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        asset = upload.json()
        self.assertEqual(asset["url"], f"http://testserver/v1/files/{asset['id']}")

    def test_file_asset_syncs_through_generic_protocol(self) -> None:
        # UPSERT: 附件元数据经通用同步协议入服务端, 其他设备可从 snapshot/changes 拉取
        upsert = self.client.post(
            "/v1/sync/operations",
            headers=self.headers,
            json={
                "operation_id": "op-file-asset-1",
                "entity_type": "file_asset",
                "entity_id": "synced-file-1",
                "operation": "UPSERT",
                "sequence": 1,
                "schema_version": 1,
                "payload": {
                    "workspace_id": "workspace-1",
                    "conversation_id": "conversation-1",
                    "message_id": "message-1",
                    "display_name": "report.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": 1024,
                    "url": "http://testserver/v1/files/synced-file-1",
                },
            },
        )
        self.assertEqual(upsert.status_code, 200, upsert.text)
        self.assertTrue(upsert.json()["applied"])

        snapshot = self.client.get("/v1/sync/snapshot", headers=self.headers).json()
        file_assets = [entity for entity in snapshot["entities"] if entity["entity_type"] == "file_asset"]
        self.assertEqual(len(file_assets), 1)
        self.assertEqual(file_assets[0]["entity_id"], "synced-file-1")
        self.assertEqual(file_assets[0]["payload"]["display_name"], "report.pdf")
        self.assertEqual(file_assets[0]["payload"]["message_id"], "message-1")
        self.assertEqual(file_assets[0]["payload"]["url"], "http://testserver/v1/files/synced-file-1")

        # 服务端真实落库(带 url 列)
        import asyncio

        from app.db import FileAsset, SessionLocal

        async def verify_row() -> None:
            async with SessionLocal() as session:
                record = await session.get(FileAsset, "synced-file-1")
                self.assertIsNotNone(record)
                assert record is not None
                self.assertEqual(record.url, "http://testserver/v1/files/synced-file-1")
                self.assertEqual(record.display_name, "report.pdf")

        asyncio.run(verify_row())

        # DELETE: 删即为胜, snapshot 中不再出现
        delete = self.client.post(
            "/v1/sync/operations",
            headers=self.headers,
            json={
                "operation_id": "op-file-asset-2",
                "entity_type": "file_asset",
                "entity_id": "synced-file-1",
                "operation": "DELETE",
                "sequence": 1,
                "schema_version": 1,
                "payload": {},
            },
        )
        self.assertEqual(delete.status_code, 200, delete.text)
        snapshot = self.client.get("/v1/sync/snapshot", headers=self.headers).json()
        self.assertNotIn(
            "synced-file-1",
            [entity["entity_id"] for entity in snapshot["entities"] if entity["entity_type"] == "file_asset"],
        )
