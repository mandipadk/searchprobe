"""Domain / content type verifier."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from aris.core.models import Document
from aris.que.models import Constraint
from aris.verification.models import ConstraintStatus, VerificationResult

# Content type indicators
_CONTENT_TYPE_INDICATORS: dict[str, list[str]] = {
    "academic": ["arxiv.org", "scholar.google", ".edu", "doi.org", "pubmed", "ieee.org", "acm.org"],
    "documentation": ["docs.", "documentation", "readthedocs", "wiki", "reference"],
    "news": ["news", "bbc.", "cnn.", "reuters.", "nytimes.", "theguardian."],
    "blog": ["blog", "medium.com", "dev.to", "hashnode", "substack."],
    "code": ["github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com"],
    "forum": ["reddit.com", "stackoverflow.com", "quora.com", "discourse"],
    "video": ["youtube.com", "vimeo.com", "youtu.be"],
    "ecommerce": ["amazon.", "ebay.", "shopify.", "etsy."],
}


def verify_domain(document: Document, constraint: Constraint) -> VerificationResult:
    """Verify that a document matches a domain/content type constraint."""
    required_type = str(constraint.value).lower() if constraint.value else ""
    if not required_type:
        return VerificationResult(
            constraint_type="domain",
            constraint_description=constraint.raw_text,
            status=ConstraintStatus.UNKNOWN,
            confidence=0.5,
        )

    url = document.url.lower()
    domain = document.domain.lower()
    indicators = _CONTENT_TYPE_INDICATORS.get(required_type, [])

    for indicator in indicators:
        if indicator in url or indicator in domain:
            return VerificationResult(
                constraint_type="domain",
                constraint_description=f"Content type: {required_type}",
                status=ConstraintStatus.SATISFIED,
                confidence=0.9,
                evidence=f"URL/domain matches '{indicator}'",
            )

    # Check content for type indicators
    content_lower = (document.content or document.snippet).lower()
    content_indicators = {
        "academic": ["abstract", "methodology", "hypothesis", "peer-reviewed", "journal"],
        "news": ["breaking", "reported", "according to", "correspondent"],
        "blog": ["posted by", "comments", "subscribe", "opinion"],
    }

    text_hints = content_indicators.get(required_type, [])
    matches = sum(1 for h in text_hints if h in content_lower)
    if matches >= 2:
        return VerificationResult(
            constraint_type="domain",
            constraint_description=f"Content type: {required_type}",
            status=ConstraintStatus.SATISFIED,
            confidence=0.7,
            evidence=f"Content has {matches} indicators of '{required_type}'",
        )

    return VerificationResult(
        constraint_type="domain",
        constraint_description=f"Content type: {required_type}",
        status=ConstraintStatus.UNKNOWN,
        confidence=0.4,
    )
