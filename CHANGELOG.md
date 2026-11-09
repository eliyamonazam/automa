# گزارش تغییرات

## 2026-11-09 — M0: تنظیمات (`config.py`)
- **انجام‌شده:** ماژول `automa.config` با مدل‌های pydantic تغییرناپذیر (`AutomaConfig`، `BudgetLimits`، `ModelPrice`، `Tier`). ترتیب اولویت: پیش‌فرض‌ها ← `automa.toml` ← `.env` ← متغیرهای محیطی. کلید API فقط از `.env`/محیط خونده می‌شه و به‌صورت `SecretStr` نگهداری می‌شه؛ وجود هر کلید محرمانه (`api_key`، `token`، `secret`، …) در `automa.toml` خطا می‌ده. سقف‌های بودجه پیش‌فرض محدود و مثبت‌ان، مقادیر بی‌نهایت/NaN رد می‌شن، `per_call` نمی‌تونه از سقف روزانه/پروژه بیشتر باشه، و هر مدلی که به سطحی نگاشت شده باید قیمت داشته باشه. پیام‌های خطای اعتبارسنجی مقدار ورودی (و در نتیجه کلید) رو نشون نمی‌دن. فایل نمونه `automa.toml.example` اضافه شد.
- **فایل‌های اصلی:** `src/automa/config.py`، `tests/test_config.py`، `automa.toml.example`.
- **تست‌ها:** `pytest` ← ۳۳ تست سبز؛ `ruff check` و `ruff format --check` ← بدون خطا.
- **مشکلات:** ندارد. (پارسر `.env` حداقلی و داخلیه؛ فرمت‌های پیچیده مثل مقدار چندخطی پشتیبانی نمی‌شن.)
- **قدم بعدی:** CI ساده در GitHub Actions: اجرای `pytest` و `ruff` روی هر push (M0).

## 2026-11-08 — M0: ساختار پکیج Python
- **انجام‌شده:** اسکلت پکیج با چیدمان `src/`، فایل `pyproject.toml` (hatchling، وابستگی‌های `pydantic`/`httpx`/`typer`، extra `dev` با `pytest` و `ruff`)، تنظیمات `pytest` و `ruff`، `.gitignore` (شامل `.env` و `workspace/`)، `.env.example` با مقادیر نمونه، و یک CLI حداقلی (`automa version`).
- **فایل‌های اصلی:** `pyproject.toml`، `src/automa/__init__.py`، `src/automa/cli.py`، `tests/test_package.py`، `.gitignore`، `.env.example`.
- **تست‌ها:** `pytest` ← ۵ تست سبز؛ `ruff check` و `ruff format --check` ← بدون خطا.
- **مشکلات:** ندارد.
- **قدم بعدی:** تنظیمات (`config.py`): خوندن `.env` و `automa.toml` (M0).

## 2026-11-09 — M0: اسناد پایه
- سند ایده (`docs/VISION.md`)، معماری (`docs/ARCHITECTURE.md`)، نقشه راه (`ROADMAP.md`)، تصمیم‌ها (`docs/DECISIONS.md`) و دستورالعمل اجراهای خودکار (`CLAUDE.md`) اضافه شد.
- هنوز کدی وجود نداره.
- **قدم بعدی:** ساختار پکیج Python و تنظیمات (M0).
