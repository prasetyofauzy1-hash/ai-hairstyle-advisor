import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND), str(BACKEND.parent)]

import pytest  # noqa: E402

# Tests drop and recreate the schema, so they never use the app DATABASE_URL.
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", f"sqlite:///{BACKEND / '.test.db'}")


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    os.environ["IMAGE_STORAGE_DIR"] = str(tmp_path_factory.mktemp("images"))
    from app.config import settings
    settings.image_storage_dir = Path(os.environ["IMAGE_STORAGE_DIR"])
    from alembic import command
    from alembic.config import Config
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    from app.database.models import Base
    from app.database.session import engine
    Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
    command.upgrade(cfg, "head")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
