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
- [x] اجرای `run_command` داخل کانتینر Docker (پیکربندی): بخش `[sandbox]` در `automa.toml` (`mode = "local"|"docker"`، ایمیج، سقف‌ها)، اتصال به `default_tools`/CLI/Orchestrator/Verifier/benchmark، نمایش حالت در `automa status`، و شکست بسته وقتی `docker` خواسته شده ولی در دسترس نیست (D33)
- [x] `Dockerfile` و `docker-compose.yml` برای خود Automa: ایمیج چندمرحله‌ای با کاربر غیر root و بدون هیچ رازی، `.dockerignore`، سرویس compose سخت‌شده (فقط‌خوندنی، بدون capability، بدون socket داکر، کلید فقط از `.env`)؛ تست‌های ایستا (D34)

## M6 — اجرای موازی و Workflowهای ذخیره‌شده
- [x] اجرای موازی وظایف مستقل (هسته): `OrchestratorSettings.max_parallel` (پیش‌فرض ۱ = همون حلقه ترتیبی، سقف ۸)، اجرای هم‌زمان وظایف آماده در thread جدا با همون قدم‌ها، زمان‌بندی/بلاک/وضعیت پروژه فقط در thread اصلی، توقف اول ← زمان‌بندی تموم و وظایف در حال اجرا فقط قدم فعلی‌شون رو تموم می‌کنن؛ قفل مشترک فایل رویداد و FakeProvider امن برای thread (D35)
- [x] اجرای موازی وظایف مستقل (پیکربندی و CLI): `max_parallel` در `automa.toml` (بخش `[orchestrator]`) و `automa run --parallel N`، نمایش در خروجی `run`/`status`
- [x] اجرای موازی وظایف مستقل (تداخل فایل‌ها): وظایفی که چک‌هاشون مسیر هم‌پوشان دارن پشت سر هم زمان‌بندی می‌شن، و `write_file`/`delete_file` روی مسیری که وظیفه موازی در حال اجرای دیگه نوشته رد می‌شه (`PathClaims`، D37)
- [x] قالب Workflow (هسته): `WorkflowStore` در `<workspace>/workflows/` — ذخیره یک‌باره برنامه پروژه `done` که همه وظایفش `passed`ن (با آمار تلاش و هزینه)، و پیشنهاد قالب‌های مشابه برای هدف جدید با شباهت قطعی واژه‌ها (بدون مدل و هزینه، فارسی و انگلیسی)؛ شکست بسته روی فایل خراب/symlink (D38)
- [x] قالب Workflow (CLI): `automa workflow save <project> [--json]`، `automa workflows [--json]`، و نمایش قالب‌های مشابه در `automa new` (فقط پیشنهاد؛ بدون کلید و بدون فراخوانی مدل، D39)
- [x] قالب Workflow (استفاده): `automa new --workflow <id>` برنامه قالب رو (بعد از اعتبارسنجی با نقش‌های فعلی و سطح‌های پیکربندی‌شده) بدون فراخوانی Executive برای پروژه جدید ذخیره می‌کنه (D40)
- [x] تعریف/ویرایش Workflow توسط کاربر (هسته): `WorkflowStore.create` (قالب `user` از برنامه نوشته‌شده توسط کاربر، با همه قوانین `Plan` و بررسی نقش‌ها/سطح‌ها) و `WorkflowStore.revise` (ویرایش = قالب جدید با `parent`؛ فایل والد دست نمی‌خوره)، `history` زنجیره ویرایش‌ها؛ آمار اثبات‌شده فقط برای قالب‌های `project` (D41)
- [x] تعریف/ویرایش Workflow توسط کاربر (CLI): `automa workflow create --goal … --plan <file.json>`، `automa workflow edit <id> [--goal …] [--plan <file>]`، `automa workflow show <id> [--json | --plan-json]` (خروجی قابل ویرایش؛ D42)

## M7 — Skillها و حافظه
- [x] بارگذاری Skillها (هسته): `Skill` و `SkillRegistry.from_dir` برای پوشه‌های `<name>/SKILL.md` با frontmatter محدود (`name`، `description`، `license`) بدون وابستگی YAML، شکست بسته (symlink، فایل خراب/بزرگ/غیر UTF-8، نام ناهمخوان)، و رد `allowed-tools` (Skill هیچ ابزاری نمی‌ده؛ D43)
- [x] اتصال Skillها به نقش‌ها: فیلد `skills` در فایل نقش (حداکثر ۸)، بررسی وجود Skill در `RoleRegistry(skills=…)`، و افزودن متن Skillها به پیام سیستمی Worker بین دستورالعمل نقش و پروتکل ثابت؛ Skill گمشده یا متن بیش از ۶۰٬۰۰۰ نویسه قبل از هر فراخوانی مدل خطاست؛ Policy دست نخورد (D44)
- [x] Skillها در CLI و workspace (بارگذاری): `load_skills(workspace)` = Skillهای داخلی (`src/automa/skills/builtin/`، فعلاً خالی) + `<workspace>/skills/`؛ نام تکراری بین دو منبع خطاست؛ `new`/`run`/`bench run` قبل از هر هزینه بارگذاری و به نقش‌ها و Worker وصل می‌کنن (D45)
- [x] Skillها در CLI (نمایش): `automa skills [--json]` (نام، توضیح، منبع داخلی/workspace، نقش‌هایی که استفاده می‌کنن) و خط `skills:` در `automa status` (و کلید `skills` در `status <id> --json`)؛ فقط‌خوندنی، بدون کلید و مدل؛ Skill خراب در `skills` کد ۲ و در `status` فقط گزارش (D46)
- [x] اتصال Skill به نقش‌های داخلی از تنظیمات کاربر: `[roles.<name>] skills = [...]` در `automa.toml` (فقط کلید `skills`؛ به انتهای Skillهای نقش اضافه می‌شه، سقف ۸، نقش ناشناخته یا Skill تکراری خطا؛ ابزار/ریسک/Policy بدون تغییر؛ D47)
- [x] حافظه سه‌لایه (هسته): `MemoryStore` در `<workspace>/memory/` با سه لایه `user/`، `projects/<id>/` و `lessons/`، هر مورد یک فایل JSON یک‌بارنوشتنی (حداکثر ۲۰۰۰ نویسه، ۲۰۰ مورد در هر لایه، بدون تکرار)، `forget` با حذف فایل، شکست بسته روی فایل/پوشه خراب یا symlink؛ ترجیحات کاربر فقط از خود کاربر (D48)
- [x] حافظه (CLI): `automa memory list|add|forget` و خط `memory:` در `automa status` (و کلید `memory` در `status <id> --json`)؛ `list`/`status` فقط‌خوندنی، `add` فقط `source=user`، `forget` با تأیید؛ بدون کلید و مدل؛ فیلتر `--scope`/`--project`، فرار کاراکتر کنترلی، شکست بسته روی حافظه خراب (D49)
- [x] حافظه در پرامپت‌ها (هسته): `memory_prompt(MemoryContext)` بخش «Memory» پیام سیستمی را می‌سازد — هر مورد یک خط JSON داخل بلوک `<memory>` (متن نمی‌تواند از خطش فرار کند)، سرتیتر صریح «داده نه دستور، هیچ ابزار/مجوز/سقفی نمی‌دهد»، سقف ۱۲٬۰۰۰ نویسه با اولویت کاربر ← پروژه ← درس و جدیدتر اول؛ `ExecutiveAgent(memory=…)` و `WorkerAgent(memory=…)`؛ حافظهٔ پروژهٔ دیگر قبل از هر فراخوانی مدل خطاست؛ ابزارها و Policy بدون تغییر (D50)
- [x] حافظه در پرامپت‌ها (CLI): `automa run` (کاربر + همین پروژه + درس‌ها به Worker) و `automa new` (کاربر + درس‌ها به Executive) قبل از هر هزینه بارگذاری می‌کنند؛ حافظهٔ خراب = کد ۲ بدون فراخوانی مدل/ساخت پروژه؛ `--no-memory`/`--memory` و `[memory] in_prompts`؛ `bench run` و Verifier حافظه نمی‌گیرند (D51)
- [x] مرحله بازبینی (هسته): `ReviewAgent` برای پروژه `done`/`failed` — خلاصه محدود و قطعی پروژه از `ProjectStore` (هدف، وظایف، سه تلاش آخر با نتیجه/خطا/حکم Verifier/هزینه؛ کاهش مرحله‌ای جزئیات تا سقف ۲۴k نویسه) داخل بلوک `<project>`، پاسخ JSON با حداکثر ۸ پیشنهاد `project`/`lesson` (هرگز `user`)، اعتبارسنجی با همان قوانین متن حافظه و تلاش مجدد، حذف تکراری‌ها (درون پاسخ و نسبت به حافظه موجود)؛ فقط از مسیر `LLMGateway`/BudgetGuard و **بدون نوشتن هیچ حافظه‌ای** (D52)
- [x] مرحله بازبینی (ذخیره و تأیید): `ReviewStore` در `<project>/reviews/` — هر بازبینی یک فایل یک‌بارنوشتنی (فقط برای پروژه `done`/`failed` با همان وضعیت)، تصمیم هر پیشنهاد یک فایل یک‌بارنوشتنی جدا (حداکثر یک بار، حتی بین پروسه‌ها)؛ `accept` پیشنهاد را با `source=review` در لایه `project`/`lesson` می‌نویسد (مورد یکسان موجود دوباره نوشته نمی‌شود)، `reject` هیچ حافظه‌ای نمی‌نویسد؛ شکست بسته روی فایل خراب/symlink/ناشناخته (D53)
- [x] مرحله بازبینی (CLI، اجرا و نمایش): `automa review run <id>` (فقط پروژه `done`/`failed`؛ همه بررسی‌های رایگان قبل از هر فراخوانی مدل؛ بازبینی دوم فقط با `--again`؛ `ReviewAgent` از مسیر `LLMGateway`/BudgetGuard و هزینه به حساب همان پروژه؛ ذخیره با `ReviewStore` بدون نوشتن حافظه) و `automa review show <id> [--review <rid>] [--json]` (فقط‌خواندنی: پیشنهادهای شماره‌دار با متن کامل escape‌شده و وضعیت pending/accepted/rejected) (D54)
- [x] مرحله بازبینی (CLI، تأیید/رد): `automa review accept <id> <n>…` و `automa review reject <id> <n>…` (پیش‌فرض آخرین بازبینی، `--review <rid>`؛ نمایش متن و تأیید قبل از نوشتن حافظه، `--yes`)، و خط `review:` در `automa status` (D55)

## M8 — Router هوشمند و Financial Controller
- [x] ثبت کیفیت (هسته): `QualityLog` در `<workspace>/quality.sqlite3` — یک ردیف برای هر تلاش داوری‌شده (پروژه، وظیفه، تلاش، نقش = نوع وظیفه، سطح، مدل، وضعیت نتیجه، حکم Verifier، هزینه Worker)، ثبت یک‌باره و idempotent، `stats` برای هر (نقش، مدل): تعداد، قبولی، نرخ قبولی، هزینه کل/میانگین/به‌ازای هر قبولی؛ ثبت زنده در `Orchestrator` (در `automa run` و `bench run`) بعد از ذخیره هر Verdict و `sync_project` برای وارد کردن دوباره از `ProjectStore`؛ خطای ثبت هیچ‌وقت اجرا را عوض نمی‌کند (D56)
- [x] ثبت کیفیت (CLI): `automa quality [--role R] [--json]` (فقط‌خواندنی؛ جدول نقش × مدل با نرخ قبولی و هزینه) و `automa quality sync [<project>]` (وارد کردن تلاش‌های داوری‌شده پروژه‌های قبلی)
- [x] Router داده‌محور (هسته): `Router` خالص در `automa.router` — برای هر نقش و سطح درخواستی، از آمار `QualityLog` مدل پشت هر سطح پیکربندی‌شده، ارزان‌ترین سطح «واجد شرایط» (حداقل `min_attempts` تلاش و نرخ قبولی ≥ `min_pass_rate`) را انتخاب می‌کند؛ گران‌تر از سطح درخواستی فقط وقتی که سطح درخواستی با داده «ضعیف ثابت‌شده» باشد؛ بخش `[router]` در `automa.toml` (`enabled` پیش‌فرض خاموش)؛ هیچ فراخوانی مدل، نوشتن یا تغییری در BudgetGuard/Policy (D58)
- [x] Router داده‌محور (اتصال به Orchestrator): `Orchestrator(router=…)` سطح شروع هر وظیفه تازه (بدون تلاش و بدون سطح ذخیره‌شده) را از `Router` می‌گیرد و در state ذخیره می‌کند (ادامه اجرا دوباره مسیریابی نمی‌کند، ارتقای سطح از همان‌جا)، رویداد `router.decision`؛ سطح درخواستی پیکربندی‌نشده ← توقف `config` قبل از هر فراخوانی (D59)
- [x] Router داده‌محور (اتصال به `automa run`): وقتی `[router] enabled` (یا `--router`؛ `--no-router` خاموش می‌کند) Router از snapshot آمار `QualityLog` قبل از هر فراخوانی مدل ساخته و به Orchestrator داده می‌شود (خط `router:` در خروجی)؛ آمار خراب ← کد ۲ بدون فراخوانی؛ `bench run` مسیریابی نمی‌کند (D60)
- [x] Router داده‌محور (پیش‌نمایش): `automa quality route [--role R] [--tier T] [--json]` فقط‌خواندنی — برای هر نقش، سطح انتخابی Router، دلیل و شواهد هر سطح را نشان می‌دهد (حتی وقتی `[router] enabled` خاموش است؛ بدون کلید و بدون ساخت فایل) (D61)
- [x] Financial Controller (تخمین، هسته): `estimate_plan` خالص در `automa.finance` — برای هر وظیفهٔ باقی‌مانده بدترین حالت قطعی (تلاش‌ها × (گام‌های Worker + فراخوانی‌های Verifier) × سقف هر فراخوانی) و هزینهٔ مورد انتظار از آمار `QualityLog` با برنامهٔ ارتقای سطح Orchestrator (بدون دادهٔ کافی ← نامعلوم)، Router اختیاری، مقایسه با بودجهٔ باقی‌مانده (`assess`)؛ بدون فراخوانی مدل، بدون نوشتن (D62)
- [x] Financial Controller (تخمین، CLI): `automa estimate <id> [--json] [--max-attempts N] [--router/--no-router]` فقط‌خواندنی — بودجهٔ باقی‌ماندهٔ پروژه و امروز از Ledger (باز شدن با `mode=ro`، چیزی ساخته/نوشته نمی‌شود)، وظایف `passed`/`failed`/`blocked` و وابسته‌های وظیفهٔ شکست‌خورده کنار گذاشته، وظیفهٔ نیمه‌کاره از تلاش‌های باقی‌مانده، سطح ذخیره‌شده و شمارش ارتقا (بدون مسیریابی دوباره)، ارزیابی چهارحالته؛ بدون کلید و بدون Provider (D63)
- [x] Financial Controller (تخمین در `automa run`): خط `estimate:` در خروجی `automa run` قبل از اولین فراخوانی مدل، با همان ورودی‌های `automa estimate` (snapshot آمار، Router همان اجرا، تنظیمات همان اجرا)؛ فقط اطلاع، بدون توقف یا تغییر BudgetGuard؛ رویداد `finance.estimate` برای مقایسه در گزارش (D64)
- [x] Financial Controller (گزارش، هسته): `spending_report` خالص در `automa.finance.report` — خرج واقعی از ردیف‌های Ledger به تفکیک پروژه/روز UTC/مدل/نقش (نقش از نقشهٔ `(پروژه، وظیفه) → نقش`)، سطل‌های `no_project`/`no_task`/`unknown_role`، فیلتر پروژه و بازهٔ روز، هر تفکیک دقیقاً برابر جمع کل؛ بدون فراخوانی مدل و بدون نوشتن (D65)
- [x] Financial Controller (گزارش، CLI): `automa finance report [--json] [--project ID] [--since D] [--until D]` فقط‌خواندنی — Ledger با `read_only=True`، نقشهٔ نقش از پلن پروژه‌ها، بدون کلید؛ پلن ناخوانا فقط هشدار، پروژهٔ ناموجود/روز نامعتبر کد ۱، Ledger خراب کد ۲ (D66)
- [x] Financial Controller (مقایسه با تخمین، هسته): `compare_estimate` خالص در `automa.finance.compare` — آخرین رویداد `finance.estimate` پروژه (اعتبارسنجی‌شده؛ رویداد خراب ← `invalid_estimate` بدون برگشت به تخمین قدیمی‌تر) در برابر خرج Ledger از همان لحظه: وظایف تحت پوشش در برابر بدترین حالت، وظایف با مورد انتظار معلوم در برابر مورد انتظار، بقیهٔ فراخوانی‌ها جدا؛ هفت نتیجه؛ بدون فراخوانی مدل و بدون نوشتن (D67)
- [x] Financial Controller (مقایسه با تخمین، CLI): بخش «estimate vs actual» در `automa finance report` (متن و `--json` با کلید `estimate_vs_actual`) برای هر پروژه (یا فقط `--project`)، مستقل از `--since`/`--until`؛ لاگ رویداد ناخوانا/symlink فقط هشدار (D68)
- [x] Financial Controller (مصرف غیرعادی، هسته): `detect_anomalies` خالص در `automa.finance.anomaly` — هر نمونه (فراخوانی از Ledger یا تلاش داوری‌شده از `QualityLog`) در برابر میانهٔ نمونه‌های **قبلی** همان (نوع، نقش، مدل)؛ با حداقل `min_samples` (پیش‌فرض ۵) و هزینه > `factor` (پیش‌فرض ۳) × میانه و ≥ `min_cost_nusd` (پیش‌فرض $0.001) ← ناهنجاری؛ مقایسهٔ دقیق صحیح، `event_data()` برای رویداد `finance.anomaly`؛ بدون فراخوانی مدل، بدون نوشتن، بدون توقف (D69)
- [x] Financial Controller (مصرف غیرعادی، گزارش): بخش «unusual spending» در `automa finance report` (متن و کلید `anomalies` در `--json`) — تشخیص روی کل تاریخچهٔ Ledger و `QualityLog` همهٔ پروژه‌ها، فقط نمایش با `--project`/`--since`/`--until` محدود می‌شود؛ لاگ کیفیت ناموجود ساخته نمی‌شود، خراب/symlink فقط هشدار؛ فقط‌خواندنی (D70)
- [x] Financial Controller (مصرف غیرعادی، `automa run`): رویداد `finance.anomaly` و خط هشدار برای فراخوانی‌ها/تلاش‌های همان اجرا، با همان خط پایهٔ تاریخچهٔ کامل (فقط هشدار؛ اختیار توقف همچنان با BudgetGuard) — بعد از اجرا (حتی توقف یا Ctrl+C) خط `unusual spending:` و رویداد `finance.anomaly` فقط برای نمونه‌های تازهٔ همان اجرا؛ شکست بررسی ← «unavailable» بدون تغییر کد خروج (D71)

## M9 — ابزارهای بیشتر و رابط
- [x] خوندن وب (هسته): ابزار `fetch_url` در `automa.tools.web` با سطح `external` (هر فراخوانی از Approval Gate، به هیچ نقش داخلی داده نشده) — فقط `http`/`https` بدون اطلاعات ورود، رد میزبانی که به نشانی غیرعمومی (loopback، خصوصی، link-local/metadata، رزرو) resolve شود، redirect دستی حداکثر ۳ بار با بررسی دوباره، `trust_env=False` (بدون proxy محیط، `.netrc` و کوکی)، سقف ۱MiB بدنه، فقط محتوای متنی، HTML به متن بدون اسکریپت/استایل، سرتیتر «دادهٔ نامطمئن»؛ در لاگ فقط نام میزبان (D72)
- [x] خوندن وب (اتصال): `fetch_url` در فهرست نقش `researcher` (هر فراخوانی همچنان با Approval Gate؛ نقش‌های دیگر `deny`) و نمای خوانای درخواست‌های `fetch_url` در `automa approvals`/`approve`/`--json` (میزبان به شکل ارسالی، درگاه، مسیر و هشدارها) با `describe_fetch_request` خالص (D73)
- [!] خوندن وب (دامنه‌های مجاز): فهرست دامنه‌هایی در `automa.toml` که `fetch_url` به آن‌ها بدون تأیید جداگانه برسد — این یک استثنا روی Approval Gate است؛ تصمیم با صاحب مخزن (آیا اصلاً بخواهیم؟ فقط برای `researcher`؟ query مجاز؟)
- [x] جست‌وجوی وب (هسته): ابزار `web_search` در `automa.tools.search` با سطح `external` (هر فراخوانی از Approval Gate؛ نه در `default_tools()` و نه در هیچ نقش) — `SearchProvider` قابل‌تعویض و `FakeSearchProvider` برای تست؛ بدون Provider هر فراخوانی رد؛ query بدون کاراکتر کنترلی/قالب و حداکثر ۴۰۰ نویسه (قبل از Gate)، `max_results` ۱ تا ۱۰؛ فقط نتیجه‌های `http`/`https` بدون اطلاعات ورود، بدون تکرار، عنوان/خلاصهٔ پاک‌شده و کوتاه؛ متن خطای سرویس هرگز به مدل نمی‌رسد؛ در لاگ فقط نام سرویس و طول query (D74)
- [!] جست‌وجوی وب (سرویس واقعی و اتصال): انتخاب سرویس جست‌وجو (مثلاً Brave Search، Tavily، SerpAPI یا SearXNG خودمیزبان)، کلید آن در متغیر محیطی، بخش `[web_search]` در `automa.toml`، دادن `web_search` به `researcher` و نمای درخواست در `automa approvals` — نیاز به تصمیم صاحب مخزن (کدام سرویس، هزینه/سهمیه، و کلید)
- [ ] ابزارهای GitHub (فقط‌خواندنی اول): خواندن issue/PR/فایل از API عمومی GitHub با همان محافظ‌های `fetch_url`
- [ ] رابط وب ساده برای وضعیت پروژه‌ها و تأییدها
- [ ] اعلان (تلگرام یا ایمیل) برای تأییدها و گزارش‌ها

## M10 — تکامل خودکار
- [ ] Automa روی مخزن خودش به‌عنوان یک پروژه کار کنه (پیشنهاد و پیاده‌سازی موردهای ROADMAP)
- [ ] ساخت خودکار Skill با شرط عملکرد بهتر روی benchmark
