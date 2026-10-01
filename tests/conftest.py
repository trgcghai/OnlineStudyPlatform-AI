from app.core.config import get_ai_settings
from app.main import app


class NoTokenSettings:
    ai_token = None
    llm_provider = "gemini"
    gemini_api_key = None
    gemini_model = "gemini-1.5-flash"
    gemini_timeout_seconds = 8
    dropout_model_dir = "models/dropout-risk"
    dropout_min_training_samples = 10


def pytest_runtest_setup(item):
    get_ai_settings.cache_clear()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_ai_settings] = lambda: NoTokenSettings()


def pytest_runtest_teardown(item, nextitem):
    app.dependency_overrides.clear()
    get_ai_settings.cache_clear()
