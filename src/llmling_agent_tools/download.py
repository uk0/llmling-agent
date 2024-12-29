from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

import httpx
from upath import UPath


async def download_file(
    url: str,
    target_dir: str = "downloads",
    chunk_size: int = 8192,
    verify_ssl: bool = False,  # For testing, in prod should be True
) -> str:
    """Download a file and return status information."""
    start_time = time.time()
    target_path = UPath(target_dir)
    target_path.mkdir(exist_ok=True)

    filename = UPath(urlparse(url).path).name or "downloaded_file"
    full_path = target_path / filename
    try:
        async with (
            httpx.AsyncClient(verify=verify_ssl) as client,
            client.stream("GET", url, timeout=30.0) as response,
        ):
            response.raise_for_status()

            total = (
                int(response.headers["Content-Length"])
                if "Content-Length" in response.headers
                else None
            )

            with full_path.open("wb") as f:
                size = 0
                async for chunk in response.aiter_bytes(chunk_size):
                    size += len(chunk)
                    f.write(chunk)

                    if total and (size % (chunk_size * 100) == 0 or size == total):
                        progress = size / total * 100
                        speed_mbps = (size / 1_048_576) / (time.time() - start_time)
                        msg = f"\r{filename}: {progress:.1f}% ({speed_mbps:.1f} MB/s)"
                        print(msg, end="")
                        await asyncio.sleep(0)

        print()  # New line after progress
        duration = time.time() - start_time
        size_mb = size / 1_048_576

        return f"Downloaded {filename} ({size_mb:.1f}MB) at {size_mb / duration:.1f} MB/s"

    except httpx.ConnectError as e:
        return f"Connection error downloading {url}: {e}"
    except httpx.TimeoutException:
        return f"Timeout downloading {url}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} downloading {url}"
    except Exception as e:  # noqa: BLE001
        return f"Error downloading {url}: {e!s}"
