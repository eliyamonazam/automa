# گزارش تغییرات

## 2026-11-09 — M2: مدل‌های داده (`Plan`، `Task`، `TaskResult`، `Verdict`)
- **انجام‌شده:** پکیج جدید `automa.plan` با مدل‌های pydantic سخت‌گیرانه (تغییرناپذیر، `extra="forbid"`، بدون inf/NaN): `Task` (شناسه امن برای پوشه `tasks/<id>/`، `title`، `description`، `role` به‌صورت نام داده نه enum، `tier` اختیاری، `inputs`، `depends_on`، حداقل یک `acceptance_criteria`)؛ `AcceptanceCriterion` با `check` اختیاری از نوع union تفکیک‌شده: `file_exists`، `file_contains`، `command` (یک‌خطی، کد خروج ۰–۲۵۵، timeout ۱–۶۰۰ ثانیه)؛ `Plan` (`schema_version`، `version` برای re-plan، `goal`، ۱ تا ۱۰۰ وظیفه) با اعتبارسنجی گراف: شناسه تکراری، وابستگی ناشناخته، وابستگی به خود و **چرخه** (DFS غیربازگشتی، پیام شامل مسیر چرخه) رد می‌شن؛ متدهای `topological_order`، `ready_tasks`، `dependents`. `TaskResult` (تلاش، وضعیت `completed`/`failed`/`step_limit`، فایل‌ها، سطح، مدل، قدم‌ها، هزینه nano-USD، زمان‌های UTC) و `Verdict` (`pass`/`fail`، نتیجه بررسی‌ها، ایرادها، `model_evaluated`) با قیدهای سازگاری: `pass` نمی‌تونه بررسی ردشده یا ایراد داشته باشه و `fail` حداقل یک دلیل لازم داره. همه مسیرها نسبی و نرمال می‌شن و `..`، مسیر مطلق، حرف درایو، `\` و NUL رد می‌شن. `parse_plan(text)` خروجی مدل رو (با یا بدون code fence) می‌خونه و در خطا `PlanError` با `details` خلاصه و قابل فرستادن به مدل (حداکثر ۲۰ خط، بدون تکرار ورودی) می‌ده — پایه تلاش مجدد در `ExecutiveAgent`. `ProjectStore` حالا `save_plan`/`load_plan`/`has_plan` داره: نوشتن اتمی `plan.json`، اعتبارسنجی دوباره قبل از نوشتن (حتی برای کپی‌های `model_copy`/`model_construct`)، رد symlink، رویداد `plan.saved`، و فقط نسخه بزرگ‌تر می‌تونه برنامه قبلی رو جایگزین کنه (`ProjectConflict`).
- **فایل‌های اصلی:** `src/automa/plan/{__init__,models}.py`، `src/automa/project/store.py`، `tests/test_plan_models.py`، `docs/ARCHITECTURE.md`.
- **تست‌ها:** `pytest` ← ۳۹۳ تست سبز (۱۳۴ تست جدید، شامل شناسه‌ها و مسیرهای مخرب، چرخه‌های مستقیم/غیرمستقیم و زنجیره ۱۰۰تایی، JSON نامعتبر از مدل، Verdict ناسازگار، فایل plan خراب/symlink، شکست اتمی نوشتن، و یکپارچگی `FakeProvider` + `LLMGateway` + `BudgetGuard` ← `parse_plan` ← `plan.json`)؛ `ruff check` و `ruff format --check` ← بدون خطا. Guard و Policy دست نخوردن.
- **مشکلات:** بررسی‌های `command` فقط توصیف‌ان؛ اجرای اون‌ها و اعمال سطح ریسک با Tool Layer و Verifier (M3) انجام می‌شه. وجود نقش (`roles/<role>`) هنوز بررسی نمی‌شه چون نقش‌ها در M3 ساخته می‌شن. ذخیره `TaskResult`/`Verdict` در `tasks/<id>/` با Worker/Verifier/Orchestrator (M3) اضافه می‌شه.
- **قدم بعدی:** `ExecutiveAgent`: سؤال‌های روشن‌کننده + تولید `plan.json` معتبر (M2).

## 2026-11-09 — M2: `ProjectStore`
- **انجام‌شده:** پکیج جدید `automa.project` با `ProjectStore` روی `<workspace>/projects/<project-id>/`: `create` (ساخت انحصاری پوشه + `tasks/` و `files/` + `project.json` + رویداد `project.created`؛ هیچ‌وقت چیزی رو بازنویسی نمی‌کنه)، `load`، `save`، `set_status` (با رویداد `project.status`)، `exists`، `list_ids`، `list_projects`، و مسیرهای `plan_file`/`tasks_dir`/`files_dir`. مدل `Project` (pydantic، تغییرناپذیر، `schema_version`، وضعیت `ProjectStatus`، زمان‌های UTC، `revision`). شناسه پروژه قبل از ساخت هر مسیر اعتبارسنجی می‌شه (فقط `a-z0-9-`، حداکثر ۶۴ کاراکتر) و پوشه symlink یا بیرون از ریشه رد می‌شه. نوشتن `project.json` اتمیه (فایل موقت + fsync + `os.replace`). `save` همزمانی خوش‌بینانه داره: اگه `revision` روی دیسک عوض شده باشه `ProjectConflict` می‌ده و هیچ تغییری بی‌صدا گم نمی‌شه. `ProjectEventLog` (زیرکلاس `JsonlEventLog`) رویدادهای بدون `project_id` رو با شناسه پروژه پر می‌کنه و رویداد پروژه دیگه رو رد می‌کنه؛ مستقیم به‌عنوان `EventSink` ‏`BudgetGuard` قابل استفاده‌ست. فایل خراب، نسخه schema ناشناخته یا شناسه ناهمخوان → `ProjectStoreError`.
- **فایل‌های اصلی:** `src/automa/project/{__init__,store}.py`، `tests/test_project_store.py`، `docs/ARCHITECTURE.md`.
- **تست‌ها:** `pytest` ← ۲۵۹ تست سبز (۷۳ تست جدید، شامل شناسه‌های مخرب/پیمایش مسیر، symlink، فایل‌های خراب، ساخت همزمان ۲۰ نخ با یک شناسه، ذخیره همزمان ۱۰ نخ، شکست اتمی نوشتن، و یکپارچگی `LLMGateway` + `FakeProvider` + `BudgetGuard` با `events.jsonl` پروژه)؛ `ruff check` و `ruff format --check` ← بدون خطا. Guard و Policy دست نخوردن.
- **مشکلات:** قفل `save` فقط درون یک پردازه‌ست؛ بین دو پردازه، بررسی revision و نوشتن اتمی نیستن (پنجره کوچک رقابت؛ M6). بودجه اختصاصی هر پروژه هنوز در `project.json` نیست — فعلاً سقف‌های سراسری `automa.toml` اعمال می‌شن (D13). `plan.json` و محتوای `tasks/` در مورد بعدی (مدل‌های داده) نوشته می‌شن.
- **قدم بعدی:** مدل‌های داده (`Plan`، `Task`، `TaskResult`، `Verdict`) با pydantic (M2).

## 2026-11-09 — M1: تست‌های کامل برای اینکه هیچ مسیری نتونه Guard رو دور بزنه
- **انجام‌شده:** فایل تست جدید `tests/test_guard_bypass.py` با ۴۶ تست در شش گروه: (۱) ساخت Gateway بدون Guard واقعی غیرممکنه؛ (۲) هر رد شدنی (بدون پروژه، پروژه خالی/فاصله، سقف هر فراخوانی، دامنه latch‌شده روزانه/پروژه، خطای خود Guard) قبل از Provider متوقف می‌شه و هیچ رزرو یا ثبتی باقی نمی‌ذاره؛ (۳) هر خطای Provider (حتی `KeyboardInterrupt`) رزرو رو آزاد می‌کنه و هر پاسخ موفق دقیقاً یک ثبت در Ledger داره؛ (۴) سوءاستفاده مستقیم از API ‏Guard (ثبت بدون اجازه، ثبت دوباره، آزادسازی نابجا، ثبت با برآورد دیگر، خرابی Ledger) خطا می‌ده و پول خرج‌شده رو گم نمی‌کنه؛ (۵) دو Guard روی یک Ledger و راه‌اندازی دوباره همچنان سقف رو اعمال می‌کنن؛ (۶) بررسی ساختاری کد با AST: فقط `LLMGateway.complete` Provider رو صدا می‌زنه و قبلش `authorize` رو؛ هیچ ماژولی بیرون از `automa.llm` Provider وارد نمی‌کنه؛ `BudgetGuard` تنها کلاس دارای `authorize` در کد تولیده.
- **سخت‌تر شدن کد (چهار راه دور زدن بسته شد):** `LLMGateway` حالا در ساخت بررسی می‌کنه Guard متدهای `authorize`/`record`/`release` قابل فراخوانی داشته باشه (مثلاً `CostLedger` دیگه به‌عنوان Guard پذیرفته نمی‌شه)؛ usage با توکن صفر همیشه با بدترین حالت حساب می‌شه حتی اگه درگاه هزینه کمتری (مثلاً ۰) گزارش کنه، و نبود توکن ورودی با حداکثر ورودی برآوردشده؛ اگه محاسبه هزینه بعد از پاسخ Provider خطا بده، بدترین حالت در Ledger ثبت و بعد خطا بالا می‌ره؛ `project_id` فقط‌فاصله رد می‌شه. `FakeProvider` حالا هر `BaseException` رو هم می‌تونه شبیه‌سازی کنه.
- **فایل‌های اصلی:** `tests/test_guard_bypass.py`، `src/automa/llm/gateway.py`، `src/automa/budget/guard.py`، `src/automa/llm/fake.py`.
- **تست‌ها:** `pytest` ← ۱۸۶ تست سبز (۴۶ تست جدید)؛ `ruff check` و `ruff format --check` ← بدون خطا. هیچ تست قبلی حذف یا ضعیف نشد.
- **مشکلات:** رزروهای در جریان بین دو Guard جدا (دو پردازه) مشترک نیستن؛ خرج ثبت‌شده مشترکه ولی دو فراخوانی همزمان در دو پردازه می‌تونن هر کدوم جداگانه رزرو بگیرن (M6). قیمت صفر در تنظیمات یعنی مدل رایگان و Guard فقط به هزینه گزارش‌شده درگاه تکیه می‌کنه — این تصمیم کاربره، نه راه دور زدن.
- **قدم بعدی:** `ProjectStore`: ساخت، خوندن و ذخیره پروژه و `events.jsonl` (M2).

## 2026-11-09 — M1: هشدار ۵۰٪ و ۸۰٪ بودجه در لاگ رویدادها
- **انجام‌شده:** ماژول جدید `automa.events` با `Event` (pydantic، تغییرناپذیر، زمان UTC اجباری، نوع نقطه‌دار)، رابط `EventSink`، `JsonlEventLog` (افزودن یک JSON در هر خط، ساخت پوشه‌ها، خوندن با خطا روی خط خراب) و `MemoryEventSink`. `BudgetGuard` حالا پارامتر اختیاری `events` داره: وقتی خرج واقعی برای اولین بار به ۵۰٪ و ۸۰٪ سقف روزانه یا پروژه می‌رسه رویداد `budget.warning` و برای هر توقف قطعی جدید `budget.stop` می‌سازه (هر کدوم فقط یک بار). رویدادها همیشه از `guard.events()` قابل خوندن‌ان. هشدارها هیچ فراخوانی‌ای رو رد نمی‌کنن. خطای Sink → `BudgetError` (شکست بسته) بدون از دست رفتن ثبت Ledger، آزادسازی رزرو یا latch. `CostLedger.now()` اضافه شد.
- **فایل‌های اصلی:** `src/automa/events.py`، `src/automa/budget/guard.py`، `src/automa/budget/ledger.py`، `tests/test_events.py`، `tests/test_budget_warnings.py`.
- **تست‌ها:** `pytest` ← ۱۴۰ تست سبز (۳۱ تست جدید، شامل مرزهای دقیق آستانه، جهش از صفر به ۸۵٪، روز UTC جدید، Sink خراب، و یکپارچگی `LLMGateway` + `FakeProvider` + `events.jsonl`)؛ `ruff check` و `ruff format --check` ← بدون خطا. هیچ تست قبلی حذف یا ضعیف نشد.
- **مشکلات:** وضعیت «هشدار داده‌شده» فقط در حافظه‌ست؛ بعد از راه‌اندازی دوباره ممکنه یک هشدار یک بار تکرار بشه. اتصال به `events.jsonl` هر پروژه با `ProjectStore` در M2 انجام می‌شه.
- **قدم بعدی:** تست‌های کامل برای اینکه هیچ مسیری نتونه Guard رو دور بزنه (M1).

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
