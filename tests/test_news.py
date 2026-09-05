from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import MagicMock, patch

from football_vnext.infrastructure.data_sources.news import NewsRSSAdapter


def _rss(pubdate):
    return f'''<?xml version="1.0"?><rss><channel>
    <item><title>Arsenal team news</title><link>https://example.com/a</link><pubDate>{format_datetime(pubdate)}</pubDate></item>
    </channel></rss>'''


def test_search_parses_recent_headline_and_build_text():
    now = datetime.now(timezone.utc)
    response = MagicMock(content=_rss(now - timedelta(hours=1)).encode())
    with patch("requests.Session.get", return_value=response):
        adapter = NewsRSSAdapter()
        items = adapter.search("Arsenal")
    assert len(items) == 1
    assert items[0].title == "Arsenal team news"
    assert "Arsenal team news" in adapter.build_text(items)


def test_search_filters_old_headlines():
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    response = MagicMock(content=_rss(old).encode())
    with patch("requests.Session.get", return_value=response):
        assert NewsRSSAdapter().search("Arsenal", hours=36) == []


def test_search_fail_closed_on_bad_xml():
    response = MagicMock(content=b"not xml")
    with patch("requests.Session.get", return_value=response):
        assert NewsRSSAdapter().search("Arsenal") == []
