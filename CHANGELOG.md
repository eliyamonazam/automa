# گزارش تغییرات

## 2026-11-09 — M1: `BudgetGuard`
- **انجام‌شده:** ماژول `automa.budget.guard` با `BudgetGuard` (پیاده‌سازی واقعی `CallGuard`). قبل از هر فراخوانی، برآورد بدترین هزینه با سه سقف مقایسه می‌شه: هر فراخوانی، روزانه (روز UTC، همه پروژه‌ها) و پروژه؛ مجموع «خرج‌شده در Ledger + رزروهای در جریان + درخواست جدید» نباید از سقف بیشتر بشه. رسیدن به سقف روزانه یا پروژه = **توقف قطعی** (latch): بعد از اون حتی فراخوانی خیلی کوچک هم در همون دامنه رد می‌شه تا کاربر تصمیم بگیره. رد شدن به‌خاطر سقف هر فراخوانی latch نمی‌شه (چیزی خرج نشده). خطای مشخص `BudgetExceeded` با `scope`، سقف، مقدار خرج‌شده، مقدار درخواستی و `latched`؛ خطای پایه `BudgetError`. فراخوانی بدون `project_id` رد می‌شه. `record` همیشه اول هزینه رو در Ledger می‌نویسه، بعد اگه فراخوانی بدون `authorize` بوده خطا می‌ده و اگه هزینه واقعی سقف رو پر کرده، latch می‌کنه. محاسبات با nano-USD صحیح: سقف رو به پایین، هزینه رو به بالا گرد می‌شه. Guard و Ledger thread-safe شدن.
- **تغییر در Gateway:** `guard` حالا **اجباری**ه (`LLMGateway` بدون Guard ساخته نمی‌شه)، و پروتکل `CallGuard` متد `release` گرفت: اگه Provider خطا بده، رزرو آزاد می‌شه.
- **فایل‌های اصلی:** `src/automa/budget/guard.py`، `src/automa/budget/ledger.py` (`today()`، قفل و اتصال چندنخی)، `src/automa/llm/gateway.py`، `tests/test_guard.py`، `tests/test_llm.py`، `tests/test_ledger.py`.
- **تست‌ها:** `pytest` ← ۱۰۹ تست سبز (۱۹ تست جدید، شامل همزمانی ۵۰ نخ و یکپارچگی با `LLMGateway` + `FakeProvider`)؛ `ruff check` و `ruff format --check` ← بدون خطا.
- **مشکلات:** وضعیت latch فقط در حافظه‌ست؛ بعد از راه‌اندازی دوباره، خرج قبلی از Ledger خونده می‌شه و همچنان جلوی عبور از سقف رو می‌گیره، ولی «توقف» به‌صورت پایدار ذخیره نمی‌شه (با Approval Gate در M5 پایدار می‌شه). همزمانی چندپردازه‌ای هنوز پوشش داده نشده (M6).
- **قدم بعدی:** هشدار ۵۰٪ و ۸۰٪ بودجه در لاگ رویدادها (M1).

## 2026-11-09 — M1: `CostLedger` با SQLite
- **انجام‌شده:** پکیج جدید `automa.budget` با `CostLedger`: ثبت هر فراخوانی (زمان UTC، روز UTC، پروژه، وظیفه، سطح، مدل درخواستی و مدل پاسخ، توکن ورودی/خروجی، هزینه واقعی، برآورد بدترین حالت، هزینه گزارش‌شده درگاه) در جدول `calls`. پول به‌صورت عدد صحیح nano-USD و همیشه با گرد کردن رو به بالا ذخیره می‌شه تا جمع هزینه‌ها هیچ‌وقت کمتر از واقعیت نشه. پرس‌وجوهای `spent_nusd`/`total_cost`/`project_cost`/`daily_cost`/`entries` با فیلتر پروژه و روز. نسخه schema با `PRAGMA user_version`؛ پایگاه داده ناشناخته یا نسخه ناشناخته رد می‌شه. قیدهای `CHECK` در خود SQLite جلوی هزینه منفی رو می‌گیرن. خطای نوشتن هیچ‌وقت بلعیده نمی‌شه. متد `record(estimate, result)` با نیمه `record` از `CallGuard` سازگاره. مسیر پیش‌فرض: `<workspace>/ledger.sqlite3`.
- **فایل‌های اصلی:** `src/automa/budget/{__init__,ledger}.py`، `tests/test_ledger.py`.
- **تست‌ها:** `pytest` ← ۹۰ تست سبز (۲۰ تست جدید، شامل یکپارچگی با `LLMGateway` و `FakeProvider`)؛ `ruff check` و `ruff format --check` ← بدون خطا.
- **مشکلات:** Ledger عمداً متد `authorize` نداره تا نتونه به‌جای Guard واقعی استفاده بشه؛ اتصال اجباری به Gateway در مورد `BudgetGuard` انجام می‌شه. فعلاً همزمانی چندپردازه‌ای بررسی نشده (اجرای موازی در M6).
- **قدم بعدی:** `BudgetGuard`: سقف پروژه، روزانه و هر فراخوانی؛ توقف قطعی + خطای مشخص (M1).

## 2026-11-09 — M1: `LLMGateway` و `FakeProvider`
- **انجام‌شده:** پکیج `automa.llm` با انواع داده مستقل از ارائه‌دهنده (`Message`، `ChatRequest`، `ChatResponse`، `Usage`، `ToolSpec`، `ToolCall`، `LLMResult`)، رابط `Provider`، ارائه‌دهنده HTTP سازگار OpenAI (`OpenAICompatProvider` با `httpx`، پشتیبانی از tool calling و `usage.cost`)، و `FakeProvider` قطعی با اسکریپت پاسخ‌ها و ثبت درخواست‌ها. `LLMGateway` سطح رو به مدل نگاشت می‌کنه، `max_tokens` رو اجباری می‌کنه، قبل از هر فراخوانی برآورد بدبینانه بدترین هزینه (`CallEstimate`) می‌سازه و از قلاب `CallGuard` اجازه می‌گیره (`authorize`) و بعد هزینه واقعی رو گزارش می‌ده (`record`). هزینه همیشه با قیمت مدلِ درخواست‌شده حساب می‌شه؛ هزینه گزارش‌شده درگاه فقط می‌تونه هزینه رو بالا ببره؛ نبود usage = محاسبه با بدترین حالت. خطاهای HTTP/شبکه به `ProviderError` (با `retryable`) تبدیل می‌شن و کلید API هیچ‌وقت در پیام خطا نمیاد.
- **فایل‌های اصلی:** `src/automa/llm/{types,provider,fake,openai_compat,gateway}.py`، `tests/test_llm.py`.
- **تست‌ها:** `pytest` ← ۷۰ تست سبز (۳۰ تست جدید، همه با FakeProvider یا `httpx.MockTransport`، بدون شبکه)؛ `ruff check` و `ruff format --check` ← بدون خطا.
- **مشکلات:** `guard` فعلاً اختیاریه چون `BudgetGuard` هنوز وجود نداره؛ در مورد `BudgetGuard` باید اجباری بشه. تلاش مجدد خودکار برای خطاهای `retryable` هنوز پیاده نشده (عمداً، تا با بودجه هماهنگ بشه).
- **قدم بعدی:** `CostLedger` با SQLite: ثبت هر فراخوانی (M1).

## 2026-11-09 — M0: CI در GitHub Actions
- **انجام‌شده:** workflow `.github/workflows/ci.yml` روی هر push و pull request: نصب `.[dev]` و اجرای `ruff check`، `ruff format --check` و `pytest` روی Python 3.11، 3.12 و 3.13. توکن فقط‌خواندنی (`contents: read`)، بدون هیچ secret یا کلید API، سقف زمانی ۱۰ دقیقه، لغو اجراهای قدیمی همان شاخه. تست `tests/test_ci_workflow.py` تضمین می‌کنه workflow همه بررسی‌ها رو اجرا کنه و هیچ‌وقت به secret/کلید دسترسی نداشته باشه.
- **فایل‌های اصلی:** `.github/workflows/ci.yml`، `tests/test_ci_workflow.py`.
- **تست‌ها:** `pytest` ← ۴۰ تست سبز؛ `ruff check` و `ruff format --check` ← بدون خطا.
- **مشکلات:** اجرای واقعی workflow فقط بعد از push روی GitHub قابل مشاهده‌ست.
- **قدم بعدی:** `LLMGateway` با API سازگار OpenAI + `FakeProvider` برای تست (M1).

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
