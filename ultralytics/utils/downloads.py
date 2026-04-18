# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Utility functions for downloading files and assets."""

import urllib
from pathlib import Path

import requests

from ultralytics.utils import LOGGER


def is_url(url, check=True):
    """Check if string is URL and optionally check if URL exists."""
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])
        if check:
            with urllib.request.urlopen(url) as response:
                return response.getcode() == 200
        return True
    except Exception:
        return False


def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    retry=3,
    min_bytes=1e0,
    progress=True,
):
    """
    Download files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file. Defaults to None.
        dir (str, optional): The directory to save the downloaded file. Defaults to None.
        unzip (bool, optional): Whether to unzip the downloaded file. Defaults to True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Defaults to False.
        retry (int, optional): Number of times to retry the download in case of failure. Defaults to 3.
        min_bytes (float, optional): Minimum file size in bytes for the download to be considered successful. Defaults to 1E0.
        progress (bool, optional): Whether to display a progress bar during the download. Defaults to True.
    """
    if ".drive.google.com" in url:
        return gdrive_download(url, file)

    f = Path(dir or ".") / (file or Path(url).name)  # target filepath
    if not f.is_file() or f.stat().st_size < min_bytes:
        desc = f"Downloading {url} to {f}"
        LOGGER.info(f"{desc}...")
        f.parent.mkdir(parents=True, exist_ok=True)
        for i in range(retry + 1):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(f, "wb") as f_opened:
                    for chunk in response.iter_content(chunk_size=8192):
                        f_opened.write(chunk)
                if f.stat().st_size < min_bytes:
                    raise ValueError(f"Downloaded file is too small: {f.stat().st_size} bytes")
                break
            except Exception as e:
                if i >= retry:
                    raise ConnectionError(f"Failed to download {url}") from e
                LOGGER.warning(f"Download failed, retrying {i + 1}/{retry}: {e}")

    if unzip and f.suffix in (".zip", ".tar", ".gz"):
        unzip_file(f)
        if delete:
            f.unlink()

    return f


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    """
    Unzip a *.zip file to the specified path, excluding files containing strings in the exclude list.

    Args:
        file (str): Path to the zip file.
        path (str, optional): Directory to unzip into. Defaults to same directory as file.
        exclude (tuple): Filename patterns to exclude.

    Returns:
        Path: Directory where files were extracted.
    """
    import zipfile

    file = Path(file)
    path = Path(path or file.parent)
    with zipfile.ZipFile(file, "r") as zip_ref:
        names = [n for n in zip_ref.namelist() if not any(ex in n for ex in exclude)]
        zip_ref.extractall(path, members=names)
    LOGGER.info(f"Unzipped {file} to {path}")
    return path


def gdrive_download(id="", file="tmp.zip"):
    """Download a file from Google Drive."""
    t = time.time()
    file = Path(file)
    cookie = Path("cookie")  # gdrive cookie file
    LOGGER.info(f"Downloading {file} from Google Drive...")
    s = f'curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={id}" > /dev/null'
    r = os.system(s)
    if r != 0:
        LOGGER.warning(f"Google Drive download failed with status {r}")
    return file
