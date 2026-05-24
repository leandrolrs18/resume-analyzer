from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import ApplicationError

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
}

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def validate_uploads(files: list[UploadFile], settings: Settings) -> None:
    if not files:
        raise ApplicationError("At least one file must be provided", status_code=422)
    if len(files) > settings.max_files:
        raise ApplicationError(
            f"Maximum of {settings.max_files} files per request",
            status_code=413,
        )
    for upload in files:
        extension = Path(upload.filename or "").suffix.lower()
        if upload.content_type not in ALLOWED_MIME_TYPES or extension not in ALLOWED_EXTENSIONS:
            raise ApplicationError(
                "Unsupported file type",
                status_code=415,
                details={"filename": upload.filename, "content_type": upload.content_type},
            )
