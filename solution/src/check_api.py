"""Check the reference API without starting a server or changing its request log."""

import asyncio
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from . import api


async def check_api() -> None:
    """Exercise health, prediction, and input validation through the ASGI app."""
    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }

    # Keep smoke checks out of the monitoring request history.
    with TemporaryDirectory() as directory:
        api.REQUEST_LOG_PATH = Path(directory) / "requests.csv"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            prediction = await client.post("/predict", json=payload)
            assert prediction.status_code == 200
            assert prediction.json() == {"prediction": "setosa"}

            invalid = await client.post(
                "/predict", json={**payload, "petal_length_cm": -1}
            )
            assert invalid.status_code == 422

        with api.REQUEST_LOG_PATH.open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 1

    print("API checks passed: health, prediction, HTTP 422, and request logging.")


if __name__ == "__main__":
    asyncio.run(check_api())
