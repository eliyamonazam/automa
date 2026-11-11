# نقشه راه

هر اجرای خودکار توسعه، **اولین مورد تیک‌نخورده** رو برمی‌داره. هر مورد باید در یک اجرا قابل انجام باشه؛ اگه بزرگ‌تر بود، اول به موردهای کوچک‌تر شکسته می‌شه.

علامت‌ها: `[ ]` انجام‌نشده · `[x]` انجام‌شده · `[~]` نیمه‌کاره (توضیح در CHANGELOG) · `[!]` نیاز به تصمیم/اقدام کاربر

## M0 — پایه پروژه
- [x] سند ایده، معماری، نقشه راه، دستورالعمل اجراهای خودکار
- [x] ساختار پکیج Python (`pyproject.toml`، `src/automa/`، `tests/`)، `.gitignore`، `.env.example`
- [x] تنظیمات (`config.py`): خوندن `.env` و `automa.toml` (base_url، کلید، نگاشت سطح به مدل، قیمت‌ها، سقف‌ها)
- [x] CI ساده در GitHub Actions: اجرای `pytest` و `ruff` روی هر push

## M1 — درگاه مدل و لایه مالی
- [x] `LLMGateway` با API سازگار OpenAI + `FakeProvider` برای تست
- [x] `CostLedger` با SQLite: ثبت هر فراخوانی
- [x] `BudgetGuard`: سقف پروژه، روزانه و هر فراخوانی؛ توقف قطعی + خطای مشخص
- [x] هشدار ۵۰٪ و ۸۰٪ بودجه در لاگ رویدادها
- [x] تست‌های کامل برای اینکه هیچ مسیری نتونه Guard رو دور بزنه

## M2 — ذخیره پروژه و برنامه‌ریزی
- [x] `ProjectStore`: ساخت، خوندن و ذخیره پروژه و `events.jsonl`
- [x] مدل‌های داده (`Plan`، `Task`، `TaskResult`، `Verdict`) با pydantic
- [x] `ExecutiveAgent`: سؤال‌های روشن‌کننده + تولید `plan.json` معتبر (با اعتبارسنجی و تلاش مجدد برای JSON نامعتبر)
- [x] CLI: `automa new "<هدف>"`، `automa status`

## M3 — اجرا و راستی‌آزمایی (اولین حلقه کامل)
- [x] Tool Layer (پایه): `Tool` با سطح ریسک، `PermissionPolicy`، `ToolExecutor`، `Workspace` محدود به `files/`، ابزارهای `read_file`، `write_file`، `list_dir`
- [x] Tool Layer: `run_command` (سطح `execute`، بدون shell، cwd داخل `files/`، timeout، محدودیت خروجی، محیط پاک بدون کلیدها)
- [x] نقش‌ها به‌صورت فایل در `roles/`: `researcher`، `architect`، `developer`، `writer`
- [x] `WorkerAgent`: حلقه tool-use تا اتمام وظیفه یا رسیدن به سقف قدم
- [x] `VerifierAgent`: بررسی‌های قطعی + ارزیابی مدلی جدا از Worker
- [x] Orchestrator (پایه): ذخیره تاریخچه تلاش‌ها و وضعیت وظایف در `tasks/<id>/` (`TaskResult`، `Verdict`، `TaskState`) در `ProjectStore`
- [x] `Orchestrator`: حلقه کامل با اصلاح (حداکثر N تلاش) و ارتقای سطح مدل
- [x] CLI: `automa run <project>` با قابلیت ادامه بعد از توقف
- [x] تست انتها‌به‌انتها با FakeProvider

## M4 — اولین آزمون واقعی
- [x] `benchmarks/` (پایه): قالب `<id>.toml`، بارگذاری بسته، ۳ هدف نمونه (ابزار CLI، صفحه وب ایستا، گزارش تحقیقی کوتاه) با معیار موفقیت فقط قطعی، و ارزیابی معیارها روی `files/` از Tool Layer
- [x] اجراکننده benchmark: `automa bench run <id>` (ساخت پروژه با `answer` و سقف `budget_usd` benchmark، اجرا با Orchestrator، ارزیابی) و ثبت نتیجه (موفقیت، امتیاز، هزینه، تلاش‌ها) در `benchmarks/results/`؛ تست با FakeProvider
- [!] اجرای واقعی benchmark با کلید API کاربر (نیاز به کلید و بودجه)
- [x] ابزار تحلیل شکست‌ها: `automa bench report` (فقط‌خوندنی) — نرخ قبولی، امتیاز، هزینه، دلایل توقف، دسته علت شکست و معیارهایی که بیشتر رد می‌شن، از روی `benchmarks/results/`
- [!] تحلیل شکست‌های اجرای واقعی با `automa bench report` و ثبت در `docs/DECISIONS.md` (بعد از اجرای واقعی benchmark)

## M5 — Approval Gate و ایزوله‌سازی
- [x] Approval Gate (هسته): ثبت درخواست تأیید در `approvals/` پروژه هنگام توقف روی ابزار `external`، تصمیم یک‌باره (تأیید/رد)، و اجرای فقط همون فراخوانی (همون وظیفه، ابزار و آرگومان‌ها) حداکثر یک بار بعد از تأیید؛ رد ← خبر به مدل بدون اجرا
- [x] Approval Gate (CLI): `automa approvals [<project>]` (فهرست درخواست‌ها با آرگومان‌ها)، `automa approve <project> <id> [--note]` / `automa reject <project> <id> [--note]`، نمایش درخواست در انتظار در `automa status` و خروجی `automa run`
- [x] Approval Gate (اقدامات برگشت‌ناپذیر): علامت `irreversible` روی ابزار که حتی در سطح `write`/`execute` از Approval Gate رد بشه، و ابزار برگشت‌ناپذیر `delete_file` (بازنویسی با `write_file` عمداً gate نشد؛ D31)
- [x] اجرای `run_command` داخل کانتینر Docker (هسته): `DockerSandbox` (ایمیج، سقف حافظه/CPU/پروسه) و `RunCommand(sandbox=…)` با کانتینر یک‌بارمصرف بدون شبکه، فایل‌سیستم فقط‌خوندنی، بدون capability، کاربر غیر root، تنها mount پوشه `files/`؛ حذف کانتینر در timeout؛ بدون Docker ← خطا و هیچ اجرایی روی میزبان (D32)
- [ ] اجرای `run_command` داخل کانتینر Docker (پیکربندی): بخش `[sandbox]` در `automa.toml` (`mode = "local"|"docker"`، ایمیج، سقف‌ها)، اتصال به `default_tools`/CLI/Orchestrator/Verifier/benchmark، نمایش حالت در `automa status`، و شکست بسته وقتی `docker` خواسته شده ولی در دسترس نیست
- [ ] `Dockerfile` و `docker-compose.yml` برای خود Automa

## M6 — اجرای موازی و Workflowهای ذخیره‌شده
- [ ] اجرای موازی وظایف مستقل
- [ ] ذخیره برنامه‌های موفق به‌عنوان قالب Workflow و پیشنهادشون برای اهداف مشابه
- [ ] امکان تعریف/ویرایش Workflow توسط کاربر

## M7 — Skillها و حافظه
- [ ] بارگذاری Skillها با قالب `SKILL.md` و اتصال به نقش‌ها
- [ ] حافظه سه‌لایه (کاربر، پروژه، درس‌های عمومی) + مرحله بازبینی بعد از هر پروژه

## M8 — Router هوشمند و Financial Controller
- [ ] ثبت کیفیت (نتیجه Verifier) کنار هزینه هر مدل برای هر نوع وظیفه
- [ ] Router داده‌محور: انتخاب ارزان‌ترین مدلی که برای آن نوع وظیفه نرخ قبولی کافی داره
- [ ] Financial Controller Agent: تخمین هزینه قبل از شروع، گزارش، تشخیص مصرف غیرعادی

## M9 — ابزارهای بیشتر و رابط
- [ ] ابزارهای GitHub، جست‌وجو و خوندن وب
- [ ] رابط وب ساده برای وضعیت پروژه‌ها و تأییدها
- [ ] اعلان (تلگرام یا ایمیل) برای تأییدها و گزارش‌ها

## M10 — تکامل خودکار
- [ ] Automa روی مخزن خودش به‌عنوان یک پروژه کار کنه (پیشنهاد و پیاده‌سازی موردهای ROADMAP)
- [ ] ساخت خودکار Skill با شرط عملکرد بهتر روی benchmark
