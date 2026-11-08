# گزارش تغییرات

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
