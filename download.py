"""Download Orbit Wars starter kit into ./starter_kit."""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DEST = Path(__file__).parent / "starter_kit"


def main() -> None:
    DEST.mkdir(exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files("orbit-wars", path=str(DEST), quiet=False)

    # competition_download_files writes a single zip; unzip it in place.
    for zip_path in DEST.glob("*.zip"):
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DEST)
        zip_path.unlink()

    print(f"Starter kit extracted to: {DEST}")


if __name__ == "__main__":
    main()
