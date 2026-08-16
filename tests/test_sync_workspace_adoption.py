"""Sync regression tests for cross-user workspace id collisions.

Workspace ids are client-chosen constants (the default workspace is literally
"inbox"), so after a device identity change (or DB restore) the same id can still
be owned by an orphaned user row. The new identity's sync UPSERT used to crash
with `sqlite3.IntegrityError: UNIQUE constraint failed: workspaces.id`.
`apply_sync_payload` now reclaims the orphaned row for the current user; every
other entity type keeps its (id, user_id) scoping — no cross-user takeover.
"""

import asyncio
import os
import unittest

from tests import _test_db

os.environ["DATABASE_URL"] = _test_db.DATABASE_URL

from fastapi.testclient import TestClient

from app.db import Conversation, SessionLocal, Workspace
from app.main import create_app

WORKSPACE_PAYLOAD = {"id": "inbox", "name": "Personal", "system_prompt": "", "is_archived": 0}


def register_device(client: TestClient, device_id: str) -> dict[str, str]:
    response = client.post(
        "/v1/auth/device",
        json={"device_id": device_id, "display_name": "Test device"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def sync_upsert(
    client: TestClient,
    headers: dict[str, str],
    operation_id: str,
    entity_type: str,
    entity_id: str,
    payload: dict[str, object],
) -> None:
    response = client.post(
        "/v1/sync/operations",
        headers=headers,
        json={
            "operation_id": operation_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operation": "UPSERT",
            "sequence": 1,
            "payload": payload,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["applied"] is True


class SyncWorkspaceAdoptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def test_workspace_id_orphaned_by_another_user_is_reclaimed(self) -> None:
        legacy_headers = register_device(self.client, "legacy-uuid-device")
        current_headers = register_device(self.client, "fatai-device-local-default")

        # Old client version (device id = random uuid) syncs the fixed default
        # workspace id first.
        sync_upsert(
            self.client, legacy_headers, "legacy:ws:1", "workspace", "inbox", WORKSPACE_PAYLOAD
        )

        # The current client syncs the same constant id. Previously this raised
        # UNIQUE constraint failed: workspaces.id because the lookup is scoped to
        # (id, user_id) and the INSERT hit the global unique id.
        sync_upsert(
            self.client, current_headers, "current:ws:1", "workspace", "inbox", WORKSPACE_PAYLOAD
        )

        async def verify() -> None:
            async with SessionLocal() as session:
                workspace = await session.get(Workspace, "inbox")
                self.assertIsNotNone(workspace)
                assert workspace is not None
                self.assertEqual(workspace.user_id, "fatai-device-local-default")
                self.assertEqual(workspace.name, "Personal")

        asyncio.run(verify())

    def test_delete_stays_scoped_to_the_owner(self) -> None:
        owner_headers = register_device(self.client, "owner-device-0001")
        stranger_headers = register_device(self.client, "stranger-device-01")
        sync_upsert(
            self.client,
            owner_headers,
            "owner:ws:1",
            "workspace",
            "keep",
            {"id": "keep", "name": "Keep", "system_prompt": "", "is_archived": 0},
        )

        # Delete-wins only applies to the sender's own records: a DELETE for an id
        # owned by someone else must not remove their row.
        response = self.client.post(
            "/v1/sync/operations",
            headers=stranger_headers,
            json={
                "operation_id": "stranger:ws:delete",
                "entity_type": "workspace",
                "entity_id": "keep",
                "operation": "DELETE",
                "sequence": 99,
                "payload": {"id": "keep"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["applied"])

        async def verify() -> None:
            async with SessionLocal() as session:
                workspace = await session.get(Workspace, "keep")
                self.assertIsNotNone(workspace)
                assert workspace is not None
                self.assertEqual(workspace.user_id, "owner-device-0001")

        asyncio.run(verify())

    def test_non_workspace_entities_are_not_reclaimed_across_users(self) -> None:
        # Conversations use client-generated uuids, so a cross-user id collision is
        # never the same logical client; silently adopting it would be a takeover
        # vector. Only the workspace constant ids are reclaimed.
        owner_headers = register_device(self.client, "conv-owner-device-01")
        stranger_headers = register_device(self.client, "conv-stranger-device")
        sync_upsert(
            self.client,
            owner_headers,
            "conv-owner:ws:1",
            "workspace",
            "conv-ws",
            {"id": "conv-ws", "name": "W", "system_prompt": "", "is_archived": 0},
        )
        conversation_payload = {
            "id": "conv-1",
            "workspace_id": "conv-ws",
            "title": "T",
            "provider_type": "OpenAI",
            "model": "gpt-4o-mini",
            "is_pinned": False,
            "is_archived": False,
        }
        sync_upsert(
            self.client, owner_headers, "conv-owner:cv:1", "conversation", "conv-1", conversation_payload
        )

        # Surface the server's 500 instead of re-raising the IntegrityError.
        client = TestClient(create_app(), raise_server_exceptions=False)
        client.__enter__()
        try:
            stolen = client.post(
                "/v1/sync/operations",
                headers=stranger_headers,
            json={
                "operation_id": "conv-stranger:cv:1",
                "entity_type": "conversation",
                "entity_id": "conv-1",
                "operation": "UPSERT",
                "sequence": 1,
                "payload": conversation_payload,
            },
            )
            # The global unique id constraint rejects the INSERT instead of taking
            # the conversation away from its owner.
            self.assertEqual(stolen.status_code, 500, stolen.text)
        finally:
            client.__exit__(None, None, None)

        async def verify() -> None:
            async with SessionLocal() as session:
                conversation = await session.get(Conversation, "conv-1")
                self.assertIsNotNone(conversation)
                assert conversation is not None
                self.assertEqual(conversation.user_id, "conv-owner-device-01")

        asyncio.run(verify())


if __name__ == "__main__":
    unittest.main()
