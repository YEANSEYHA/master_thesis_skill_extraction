from dataclasses import dataclass
from enum import Enum
from typing import Optional


class URLStatus(Enum):
    VALID = "valid"
    DNS_ERROR = "dns_error"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    SSL_ERROR = "ssl_error"
    REDIRECT_LOOP = "redirect_loop"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ValidationResult:
    url: str
    status: URLStatus
    status_code: Optional[int] = None
    final_url: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status == URLStatus.VALID


@dataclass
class VerificationResult:
    """Result from AI agent verification."""
    url: str
    is_job_site: bool
    confidence: float
    site_type: str
    reason: str
    features_found: list[str]
