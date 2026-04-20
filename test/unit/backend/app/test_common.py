from __future__ import annotations

import unittest
from datetime import datetime, timezone

from backend.app.domain.common import utc_now


class UtcNowTest(unittest.TestCase):
    def test_utc_now_returns_datetime(self) -> None:
        now = utc_now()
        self.assertIsInstance(now, datetime)

    def test_utc_now_has_utc_timezone(self) -> None:
        now = utc_now()
        self.assertEqual(now.tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main()