import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Union

log = logging.getLogger(__name__)


def write_archived(folder: Union[str, Path], filename: str, write: Callable[[Path], None],
                   update_latest: bool = True) -> Path:
    """Write ``folder/archive/<YYYYMMDD>_<filename>`` through ``write(path)``.

    With ``update_latest``, ``folder/default.<ext>`` becomes a relative symlink to the
    new file, which is what the loaders read. Sysid results are written this way so
    every identification stays on disk and the active one is a symlink flip.
    """
    archive = Path(folder) / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    path = archive / f"{datetime.now():%Y%m%d}_{filename}"
    if path.exists():
        log.warning("%s already exists and will be overwritten", path)
    write(path)
    log.info("Saved %s", path)

    if update_latest:
        latest = Path(folder) / ("default" + path.suffix)
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(Path("archive") / path.name)
        log.info("%s now points to %s", latest, path)
    return path
