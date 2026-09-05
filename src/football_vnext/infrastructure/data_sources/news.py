"""Public news collector with no API key requirement.

Google News RSS is used only as a text source.  The statistical model never
reads headlines directly; callers may pass the collected text to the existing
bounded NewsImpactAnalyzer when an Anthropic key is configured.
"""
from __future__ import annotations
import logging
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
import requests

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NewsItem:
    title: str
    link: str
    published: datetime | None
    source: str = "Google News RSS"

class NewsRSSAdapter:
    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "football-vnext/1.0"})

    def search(self, team_name: str, hours: int = 36, limit: int = 8) -> list[NewsItem]:
        q = urllib.parse.quote(f'"{team_name}" football')
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        try:
            root = ET.fromstring(self.session.get(url, timeout=self.timeout).content)
        except Exception as exc:
            logger.warning("News RSS unavailable for %s: %s", team_name, exc)
            return []
        now = datetime.now(timezone.utc)
        out=[]
        for item in root.findall('./channel/item')[:limit]:
            title=(item.findtext('title') or '').strip()
            link=(item.findtext('link') or '').strip()
            pub=(item.findtext('pubDate') or '').strip()
            dt=None
            try:
                from email.utils import parsedate_to_datetime
                dt=parsedate_to_datetime(pub).astimezone(timezone.utc)
            except Exception: pass
            if dt is not None and (now-dt).total_seconds() > hours*3600:
                continue
            if title: out.append(NewsItem(title, link, dt))
        return out

    def build_text(self, items: list[NewsItem]) -> str:
        return "\n".join(f"- {x.title}" for x in items)
