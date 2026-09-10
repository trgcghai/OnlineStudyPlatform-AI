import importlib
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class SettingsTests(unittest.TestCase):
    def test_database_settings_load_from_env_file(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        original_cwd = Path.cwd()
        original_sys_path = list(sys.path)
        original_env = os.environ.copy()

        with TemporaryDirectory() as temp_dir:
            try:
                Path(temp_dir, ".env").write_text(
                    "DB_HOST=127.0.0.1\n"
                    "DB_PORT=5433\n"
                    "DB_USERNAME=postgres\n"
                    "DB_PASSWORD=secret\n"
                    "DB_NAME=kltn\n",
                    encoding="utf-8",
                )

                for key in ["DB_HOST", "DB_PORT", "DB_USERNAME", "DB_PASSWORD", "DB_NAME"]:
                    os.environ.pop(key, None)

                os.chdir(temp_dir)
                sys.path.insert(0, str(project_root))
                sys.modules.pop("app.core.config", None)

                config = importlib.import_module("app.core.config")
                settings = config.get_database_settings()

                self.assertEqual(settings.host, "127.0.0.1")
                self.assertEqual(settings.port, 5433)
                self.assertEqual(settings.username, "postgres")
                self.assertEqual(settings.password, "secret")
                self.assertEqual(settings.database, "kltn")
            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_sys_path
                os.environ.clear()
                os.environ.update(original_env)
                sys.modules.pop("app.core.config", None)

    def test_mongo_settings_load_from_env_file(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        original_cwd = Path.cwd()
        original_sys_path = list(sys.path)
        original_env = os.environ.copy()

        with TemporaryDirectory() as temp_dir:
            try:
                Path(temp_dir, ".env").write_text(
                    "MONGODB_URI=mongodb://root:secret@127.0.0.1:27017/kltn?authSource=admin\n",
                    encoding="utf-8",
                )

                os.environ.pop("MONGODB_URI", None)

                os.chdir(temp_dir)
                sys.path.insert(0, str(project_root))
                sys.modules.pop("app.core.config", None)

                config = importlib.import_module("app.core.config")
                settings = config.get_mongo_settings()

                self.assertEqual(
                    settings.mongodb_uri,
                    "mongodb://root:secret@127.0.0.1:27017/kltn?authSource=admin",
                )
            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_sys_path
                os.environ.clear()
                os.environ.update(original_env)
                sys.modules.pop("app.core.config", None)


if __name__ == "__main__":
    unittest.main()
