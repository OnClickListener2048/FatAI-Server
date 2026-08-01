import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db import ModelConfiguration
from app.services.errors import ServiceError


class UserModelCredentials:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model


def encrypt_api_key(api_key: str, settings: Settings) -> str:
    return _cipher(settings).encrypt(api_key.encode("utf-8")).decode("utf-8")


def decrypt_api_key(ciphertext: str, settings: Settings) -> str:
    try:
        return _cipher(settings).decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError) as error:
        raise ServiceError(
            "MODEL_CONFIGURATION_UNAVAILABLE",
            "The saved model credential cannot be decrypted. Update the model configuration.",
            503,
        ) from error


async def get_user_model_credentials(
    session: AsyncSession,
    user_id: str,
    configuration_id: str | None,
    settings: Settings,
) -> UserModelCredentials:
    query = select(ModelConfiguration).where(ModelConfiguration.user_id == user_id)
    if configuration_id:
        query = query.where(ModelConfiguration.id == configuration_id)
    else:
        query = query.where(ModelConfiguration.is_active.is_(True))
    configuration = await session.scalar(query)
    if configuration is None:
        raise ServiceError(
            "MODEL_NOT_CONFIGURED",
            "Configure and select a model provider before starting a chat.",
            503,
        )
    return UserModelCredentials(
        api_key=decrypt_api_key(configuration.api_key_ciphertext, settings),
        base_url=configuration.base_url,
        model=configuration.model,
    )


def _cipher(settings: Settings) -> Fernet:
    # This reuses the deployment secret so provider keys are encrypted at rest without placing
    # provider credentials in environment variables. Rotate JWT_SECRET only with a key migration.
    material = hashlib.sha256(settings.jwt_secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(material))
