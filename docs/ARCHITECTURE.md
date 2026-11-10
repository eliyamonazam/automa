# معماری Automa

این سند معماری فنی سیستم رو تعریف می‌کنه. هر تغییر مهم در معماری باید اینجا و در `DECISIONS.md` ثبت بشه.

## اصول پایه

1. **هسته کوچک، لایه‌های افزودنی.** قلب سیستم یک حلقه‌ست: هدف ← برنامه ← اجرا ← راستی‌آزمایی ← اصلاح. بقیه قابلیت‌ها (Router، حافظه، ساخت Skill، …) لایه‌هایی روی این حلقه‌ان.
2. **اجماع جای راستی‌آزمایی رو نمی‌گیره.** هر وظیفه معیار پذیرش (acceptance criteria) داره و ارزیابی مستقل انجام می‌شه. هر جا ممکنه، ارزیابی با چیزی بیرون از مدل انجام می‌شه: اجرای تست، اجرای برنامه، بررسی فایل.
3. **محدودیت‌های حیاتی در کد اعمال می‌شن، نه در prompt.** سقف بودجه، سطح دسترسی ابزارها و نقاط تأیید انسانی قوانین کدن، و هیچ Agentی نمی‌تونه دورشون بزنه.
4. **همه‌چیز قابل ادامه‌ست.** وضعیت پروژه بعد از هر قدم روی دیسک ذخیره می‌شه.
5. **مستقل از ارائه‌دهنده.** همه فراخوانی‌های مدل از یک درگاه واحد (`LLMGateway`) رد می‌شن.

## نمای کلی

```
                 ┌──────────────────────────┐
  کاربر ── CLI ─►│   Orchestrator (حلقه)    │◄── Approval Gate ◄── کاربر
                 └──────────┬───────────────┘
        ┌───────────────────┼─────────────────────┐
        ▼                   ▼                     ▼
  Executive Agent     Worker Agents          Verifier Agent
  (برنامه‌ریزی)        (اجرای وظایف)          (ارزیابی مستقل)
        │                   │                     │
        └─────────┬─────────┴──────────┬──────────┘
                  ▼                    ▼
            LLMGateway            Tool Layer
       (مدل‌ها + Router)      (فایل، shell، git، وب …)
                  │                    │
                  ▼                    ▼
           Budget Guard          Permission Policy
           + Cost Ledger
                  │
                  ▼
           Project Store  (state، وظایف، خروجی‌ها، لاگ)
```

## اجزا

### CLI
`automa new "<هدف>"` تنظیمات رو می‌خونه، **قبل از ساخت پروژه** نگاشت سطح Executive و کلید API رو بررسی می‌کنه (تا پروژه یتیم نمونه)، پروژه رو می‌سازه، سؤال‌های روشن‌کننده Executive رو از کاربر می‌پرسه (`--no-ask` برای رد کردن)، پاسخ‌ها رو در `clarifications.json` ذخیره و `plan.json` رو با `plan_project` می‌نویسه. Provider فقط از `automa.llm.factory.provider_from_config` ساخته و فقط به `LLMGateway` با `BudgetGuard` روی Ledger مشترک workspace و `events.jsonl` پروژه داده می‌شه. `automa status [id] [--json]` فقط‌خوندنیه: نه کلید می‌خواد، نه فراخوانی مدل داره، نه فایلی (حتی Ledger) می‌سازه. کدهای خروج ثابت: ۰ موفق، ۱ خطای کاربر، ۲ تنظیمات، ۳ توقف بودجه (پروژه `paused`)، ۴ شکست برنامه‌ریزی، ۵ خطای Provider (جزئیات: D16).

### Orchestrator
حلقه اصلی. وضعیت پروژه رو می‌خونه، وظیفه بعدی آماده (وابستگی‌هاش انجام‌شده) رو برمی‌داره، به Worker مناسب می‌ده، نتیجه رو به Verifier می‌ده، و بر اساس نتیجه وظیفه رو «انجام‌شده»، «برگشت برای اصلاح» یا «شکست» علامت می‌زنه. بعد از هر قدم state ذخیره می‌شه.

وظایف بی‌وابستگی به هم می‌تونن موازی اجرا بشن (از مرحله M4 به بعد).

### Executive Agent
- ورودی: هدف کاربر.
- ابتدا سؤال‌های روشن‌کننده (در صورت لزوم) تولید می‌کنه.
- خروجی: یک **برنامه** به شکل JSON: فهرست وظایف، هر کدوم با `id`، `title`، `role`، `inputs`، `depends_on`، `acceptance_criteria`.
- وقتی یک وظیفه چند بار شکست بخوره، Executive می‌تونه برنامه رو بازنویسی کنه (re-plan).
- **مدل‌های داده (`src/automa/plan/models.py`):** `Plan` (`schema_version`، `version`، `goal`، `tasks`) با اعتبارسنجی گراف (شناسه یکتا، وابستگی شناخته‌شده، بدون چرخه) و `topological_order`/`ready_tasks`/`dependents`؛ `Task` (`id`، `title`، `description`، `role`، `tier`، `inputs`، `depends_on`، `acceptance_criteria`)؛ `AcceptanceCriterion` با `check` اختیاری (`file_exists`، `file_contains`، `command`)؛ `TaskResult` (خروجی یک تلاش Worker) و `Verdict` (نظر Verifier، `pass`/`fail` + `checks` + `issues`، همیشه سازگار). مسیرها نسبی به `files/` و بدون `..`. `parse_plan` خروجی مدل رو اعتبارسنجی می‌کنه و `PlanError.details` بازخورد قابل فرستادن به مدل برای تلاش مجدده (جزئیات: D14).
- **پیاده‌سازی (`src/automa/agents/executive.py`):** `ExecutiveAgent` فقط با `LLMGateway` (یعنی همیشه از Guard رد می‌شه) ساخته می‌شه. `clarify(goal)` حداکثر ۵ سؤال برمی‌گردونه (فهرست خالی = بدون سؤال)؛ `plan(goal, clarifications)` یک `Plan` معتبر می‌ده. پاسخ نامعتبر (JSON خراب، گراف نامعتبر، نقش خارج از `known_roles`، پاسخ بریده‌شده با `finish_reason=length`) با فهرست خطای `PlanError.details` به مدل برمی‌گرده و حداکثر `max_attempts` (پیش‌فرض ۳، سقف ۵) بار تکرار می‌شه، وگرنه `PlanningFailed`. خطاهای Gateway (`BudgetExceeded`، `ProviderError`) هیچ‌وقت تکرار نمی‌شن. `goal` و `version` برنامه رو کد تعیین می‌کنه نه مدل. رویدادها: `executive.questions`، `executive.reply_rejected`، `executive.plan_ready`. `plan_project(agent, store, id)` برنامه رو در `plan.json` می‌نویسه و وضعیت پروژه رو مدیریت می‌کنه (`planning`؛ شکست ← `failed`؛ توقف بودجه ← `paused`) (جزئیات: D15).

### Worker Agents و نقش‌ها (Roles)
هر نقش یک فایل در `roles/` داره: دستورالعمل سیستمی، ابزارهای مجاز، سطح مدل پیش‌فرض. نقش‌ها داده‌ان، نه کد؛ پس اضافه کردن نقش جدید نیاز به تغییر کد نداره. در آینده Skillها هم به نقش‌ها متصل می‌شن.
- **پیاده‌سازی (`src/automa/roles/`):** هر نقش یک فایل TOML ‏`<name>.toml` در `src/automa/roles/builtin/` (همراه پکیج نصب می‌شه): `name` (برابر نام فایل، همون الگوی `Task.role`)، `description`، `tier` (پیش‌فرض `medium`؛ هیچ‌وقت نام مدل)، `tools`، `max_risk` (پیش‌فرض `read`؛ `external` رد می‌شه)، `max_steps` (۱ تا ۲۰۰، سقف قدم Worker)، `instructions` (پرامپت سیستمی). مدل `Role` تغییرناپذیر و `extra="forbid"`ه و `Role.policy()` یک `PermissionPolicy` می‌سازه — این Policy اجرا می‌شه، نه متن پرامپت. `RoleRegistry.from_dir`/`builtin` بارگذاری رو بسته شکست می‌ده: فایل خراب، symlink، غیرمعمولی (مثل FIFO، بدون قفل شدن)، بزرگ‌تر از ۶۴KiB، غیر UTF-8، نام ناهمخوان یا نقش تکراری کل مجموعه رو رد می‌کنه؛ با دادن ابزارها، ابزار ناشناخته یا ابزاری بالاتر از `max_risk` نقش هم خطاست (ابزار `external` مجازه ولی همیشه Approval می‌خواد). نقش‌های داخلی: `researcher`، `architect`، `writer` (فقط خوندن/نوشتن فایل، سطح `write`) و `developer` (به‌علاوه `run_command`، سطح `execute`). `automa new` نام نقش‌ها رو به‌عنوان `known_roles` به Executive می‌ده، پس برنامه با نقش ناشناخته رد و تکرار می‌شه (جزئیات: D19).
- **`WorkerAgent` (`src/automa/agents/worker.py`):** فقط با `LLMGateway` ساخته می‌شه. `run(task, role, executor, project_id=…, attempt=…, feedback=…, tier=…)` یک تلاش رو اجرا می‌کنه و `TaskResult` برمی‌گردونه. پیام سیستمی = `instructions` نقش + قرارداد کار؛ پیام کاربر = `task_message` (عنوان، توضیح، ورودی‌ها، معیارهای پذیرش با شماره و بررسی قطعی، و ایرادهای Verifier در تلاش بعدی). هر قدم یک فراخوانی مدل با specهای ابزارهای مجاز؛ tool callها فقط از `ToolExecutor` اجرا می‌شن؛ پاسخ بدون ابزار = پایان (`completed`)؛ سقف `min(role.max_steps, settings.max_steps)` ← `step_limit`؛ `content_filter` ← `failed`. Policy Executor باید دقیقاً `role.policy()` باشه. خطاهای بودجه/Provider، `ApprovalRequired` و `ToolLayerError` بالا می‌رن و تکرار نمی‌شن. رویدادهای `worker.started`/`worker.step`/`worker.finished` بدون محتوا. ذخیره نتیجه در `tasks/<id>/` با Orchestratorه (جزئیات: D20).

### Verifier Agent
- مستقل از Worker: پیام‌ها و استدلال Worker رو نمی‌بینه، فقط خروجی و معیار پذیرش رو.
- اول **بررسی‌های قطعی** اجرا می‌شن (تست‌ها، وجود فایل‌ها، اجرای دستورها). فقط اگر قبول شدن، ارزیابی مدلی انجام می‌شه.
- خروجی: `pass` یا `fail` + فهرست ایرادها که به Worker برمی‌گرده.
- **پیاده‌سازی (`src/automa/agents/verifier.py`):** `VerifierAgent` فقط با `LLMGateway` ساخته می‌شه. `verify(task, result, executor, project_id=…, tier=…)` یک `Verdict` برمی‌گردونه. Executor باید Policy **دقیقاً** `verifier_policy()` داشته باشه (`read_file`، `list_dir`، `run_command`؛ سقف `execute`؛ بدون `write_file`). ترتیب: (۱) نتیجه غیر `completed` ← `fail` فوری بدون بررسی و بدون مدل؛ (۲) `run_checks` همه بررسی‌های قطعی رو از `ToolExecutor` اجرا می‌کنه: `file_exists` (فایل با `read_file` یا پوشه با `list_dir`)، `file_contains` (نشانه بریدن `read_file` هیچ‌وقت متن مورد انتظار حساب نمی‌شه)، `command` (`shlex.split` بدون shell ← `run_command` با `timeout_s` بررسی و مقایسه `exit_code`)؛ هر بررسی ناموفق یک `Issue` با شماره معیار می‌سازه؛ (۳) فقط اگه همه قبول شدن و معیاری بدون بررسی وجود داشت، ارزیابی مدلی: پیام فقط شامل وظیفه، معیارها و محتوای فایل‌ها (فایل‌های `result.files` + مسیر بررسی‌ها، با سقف تعداد/اندازه) — **نه** خلاصه یا گفت‌وگوی Worker، و بدون ابزار. پاسخ باید برای **هر** معیار خواسته‌شده دقیقاً یک قضاوت (`passed` بولی سخت‌گیرانه + دلیل برای رد) بده؛ پاسخ نامعتبر/بریده/فیلترشده با بازخورد تکرار می‌شه (پیش‌فرض ۲ بار) وگرنه `VerificationFailed` — هیچ‌وقت `pass`. خطاهای بودجه/Provider و `ToolLayerError` بالا می‌رن. رویدادهای `verifier.checks`/`verifier.reply_rejected`/`verifier.verdict` بدون محتوا (جزئیات: D21).

### LLMGateway
- یک کلاینت سازگار با OpenAI Chat Completions. `base_url`، `api_key` و نام مدل‌ها از تنظیمات میاد؛ پس هر درگاهی (OpenRouter، درگاه‌های داخلی، یا ارائه‌دهنده مستقیم) با تغییر تنظیمات قابل استفاده‌ست.
- **سطح‌ها (tiers):** `small`، `medium`، `large`. نقش‌ها سطح می‌خوان، نه نام مدل. نگاشت سطح به مدل در تنظیماته.
- **ارتقای سطح:** اگر وظیفه‌ای با یک سطح دو بار رد شد، با سطح بالاتر تکرار می‌شه.
- هر فراخوانی، قبل از ارسال از Budget Guard اجازه می‌گیره و بعدش در Ledger ثبت می‌شه؛ Guard اجباریه و اگه Provider خطا بده رزرو آزاد می‌شه.
- یک **ارائه‌دهنده ساختگی (FakeProvider)** برای تست‌ها، تا توسعه و آزمون بدون هزینه انجام بشه.
- **پیاده‌سازی (`src/automa/llm/`):** `types.py` (انواع داده)، `provider.py` (رابط `Provider` و `ProviderError`)، `openai_compat.py`، `fake.py`، `gateway.py`. هر فراخوانی `max_tokens` اجباری داره؛ Gateway قبل از ارسال یک `CallEstimate` (بدترین هزینه ممکن) به `CallGuard.authorize` می‌ده و بعد از پاسخ `CallGuard.record` رو با هزینه واقعی صدا می‌زنه. هزینه واقعی هیچ‌وقت کمتر از محاسبه با قیمت مدل درخواستی نیست (جزئیات: D8). Gateway در سازنده هر Guard بدون `authorize`/`record`/`release` قابل فراخوانی رو رد می‌کنه؛ usage با توکن صفر همیشه با بدترین حالت حساب می‌شه؛ خطا بعد از پاسخ Provider با ثبت بدترین حالت همراهه. `tests/test_guard_bypass.py` با بررسی AST تضمین می‌کنه فقط `LLMGateway.complete` (بعد از `authorize`) Provider رو صدا بزنه (جزئیات: D12).

### Budget Guard و Cost Ledger (لایه مالی)
- **Ledger:** هر فراخوانی با زمان، پروژه، وظیفه، مدل، توکن ورودی/خروجی و هزینه در SQLite ثبت می‌شه.
  **پیاده‌سازی (`src/automa/budget/ledger.py`):** `CostLedger` در `<workspace>/ledger.sqlite3`؛ پول به‌صورت nano-USD صحیح با گرد کردن رو به بالا، روزها UTC، نسخه schema در `user_version` (جزئیات: D9).
- **سقف‌ها:** سقف هر پروژه، سقف روزانه و سقف هر فراخوانی. رسیدن به سقف = توقف قطعی و درخواست تأیید از کاربر. این قانون در کده و Agentها نمی‌تونن تغییرش بدن.
  **پیاده‌سازی (`src/automa/budget/guard.py`):** `BudgetGuard` تنها `CallGuard` تولیدیه و `LLMGateway` بدون Guard ساخته نمی‌شه. `authorize` بدترین هزینه رو رزرو می‌کنه (خرج Ledger + رزروها + درخواست ≤ سقف)، `record` هزینه واقعی رو ثبت و رزرو رو آزاد می‌کنه، `release` رزرو فراخوانی ناموفق رو آزاد می‌کنه. رسیدن به سقف روزانه/پروژه دامنه رو latch می‌کنه و خطای `BudgetExceeded` می‌ده (جزئیات: D10).
- **هشدار:** در ۵۰٪ و ۸۰٪ سقف به کاربر گزارش می‌شه.
  **پیاده‌سازی:** `BudgetGuard` بعد از هر `record`، وقتی خرج واقعی برای اولین بار به ۵۰٪ و ۸۰٪ سقف روزانه یا پروژه می‌رسه، رویداد `budget.warning` و برای هر توقف قطعی جدید رویداد `budget.stop` می‌سازه (یک بار برای هر دامنه/کلید/آستانه). رویدادها همیشه در حافظه Guard می‌مونن و اگه `EventSink` داده شده باشه (مثلاً `JsonlEventLog` روی `events.jsonl`) به اون هم فرستاده می‌شن. خطای Sink نه ثبت Ledger رو برمی‌گردونه نه latch رو، ولی به‌صورت `BudgetError` بالا میاد (جزئیات: D11).
- قیمت مدل‌ها از تنظیمات یا پاسخ درگاه (اگر هزینه رو برگردونه) خونده می‌شه.
- **Financial Controller Agent** (مرحله بعد) روی همین داده‌ها تحلیل می‌کنه: تخمین هزینه قبل از شروع، تشخیص مصرف غیرعادی، پیشنهاد مدل ارزان‌تر. ولی اختیار نهایی همیشه با Guard در کده.

### Tool Layer و Permission Policy
- ابزارها کلاس‌های ساده با نام، توضیح، پارامترها و **سطح ریسک**ان: `read`، `write`، `execute`، `external` (شبکه/انتشار/پرداخت).
- هر نقش فهرست ابزارهای مجازش رو داره.
- ابزارهای `write` و `execute` فقط داخل پوشه کاری پروژه کار می‌کنن.
- ابزارهای سطح `external` و هر اقدام برگشت‌ناپذیر از **Approval Gate** رد می‌شن: پروژه متوقف می‌شه تا کاربر تأیید کنه.
- اجرای کد در مراحل بعد داخل کانتینر Docker ایزوله انجام می‌شه.
- **پیاده‌سازی (`src/automa/tools/`):** `base.py` (`RiskLevel` مرتب: `read` < `write` < `execute` < `external`؛ کلاس پایه `Tool` با `name`/`description`/`risk`/`args_model` که در تعریف زیرکلاس بررسی می‌شن؛ `ToolArgs` سخت‌گیرانه؛ `ToolResult`؛ `spec()` شِمای تابع سازگار OpenAI)، `workspace.py` (`Workspace` روی `files/` پروژه: هر مسیر با همون قانون مسیر امن D14 اعتبارسنجی و بعد جزءبه‌جزء با `dir_fd` + `O_NOFOLLOW` باز می‌شه، پس هیچ symlinkی — حتی اگه بعد از بررسی جایگزین بشه — دنبال نمی‌شه؛ نوشتن اتمی؛ فقط فایل معمولی UTF-8؛ سقف خوندن ۲۵۶KiB و نوشتن ۱MiB)، `files.py` (`read_file`/`list_dir` سطح `read`، `write_file` سطح `write`؛ فقط از `Workspace` استفاده می‌کنن)، `policy.py` (`PermissionPolicy` تغییرناپذیر: `allowed_tools` + `max_risk` پیش‌فرض `read`؛ ابزار خارج از فهرست یا بالاتر از سقف ← `deny`؛ `external` همیشه ← `needs_approval` و `max_risk=external` اصلاً ساخته نمی‌شه)، `executor.py` (`ToolExecutor`: تنها جایی که `Tool.run` صدا زده می‌شه، همیشه بعد از `policy.decide`؛ ابزار ناشناخته، رد سیاست، آرگومان نامعتبر و خطای ابزار به‌صورت `ToolResult` ناموفق به مدل برمی‌گردن؛ `external` خطای `ApprovalRequired` می‌ده تا Orchestrator پروژه رو متوقف کنه؛ رویدادهای `tool.call`/`tool.denied`/`tool.approval_required` بدون محتوای فایل؛ خطای لاگ رویداد ← `ToolLayerError` (شکست بسته)؛ `specs()` فقط ابزارهای مجاز رو به مدل نشون می‌ده). تست ساختاری AST تضمین می‌کنه فقط `ToolExecutor.execute` ابزار اجرا کنه و قبلش `decide` صدا زده بشه (جزئیات: D17).
- **`run_command` (`tools/command.py`، سطح `execute`):** اجرای یک برنامه بدون shell با `argv` فهرستی؛ `argv[0]` نام ساده یا مسیر نسبی داخل workspace؛ `allowed_programs` اختیاری؛ `cwd` از `Workspace.dir_path`؛ محیط فقط از فهرست مجاز (`PATH`/`LANG`/`LC_ALL`/`TZ` + `HOME`=workspace) پس کلیدها به فرزند نمی‌رسن؛ stdin بسته؛ session جدا و kill کل گروه پروسه در timeout و بعد از پایان؛ سقف ۶۴KiB برای هر جریان. خروجی: `exit_code` + stdout + stderr. `Tool.audit()` فیلدهای ثبت‌شده در `tool.call` رو تعیین می‌کنه (برای این ابزار فقط برنامه و `cwd`). sandbox نیست؛ ایزوله‌سازی با Docker در M5 (جزئیات: D18).

### Project Store
```
workspace/projects/<project-id>/
  project.json      هدف، وضعیت، بودجه، تنظیمات
  plan.json         وظایف و وابستگی‌ها
  clarifications.json  سؤال‌های Executive و پاسخ کاربر (یک بار نوشته می‌شه)
  tasks/<task-id>/  ورودی، خروجی، نتیجه ارزیابی، تاریخچه تلاش‌ها
  files/            خروجی‌های واقعی پروژه (کد، سند، …)
  events.jsonl      لاگ همه رویدادها
```
لاگ رویدادها با `automa.events` نوشته می‌شه: `Event` (زمان UTC، نوع نقطه‌دار مثل `budget.warning`، `project_id`، `data`)، رابط `EventSink`، `JsonlEventLog` (یک JSON در هر خط، فقط افزودنی، خط خراب هنگام خوندن خطا می‌ده) و `MemoryEventSink`.
پیاده‌سازی در `automa.project`: `ProjectStore` (ریشه `<workspace>/projects`) با `create`/`load`/`save`/`set_status`/`list_projects`. `project.json` مدل `Project` است (`schema_version`، `id`، `goal`، `status`، `created_at`، `updated_at`، `revision`) و اتمی نوشته می‌شه؛ `save` با `revision` همزمانی خوش‌بینانه داره (`ProjectConflict`). شناسه پروژه فقط `a-z0-9-` و پوشه پروژه نمی‌تونه symlink یا بیرون از ریشه باشه. `store.events(id)` یک `ProjectEventLog` برمی‌گردونه که `project_id` رویدادها رو اجباری می‌کنه و به‌عنوان `EventSink` به `BudgetGuard` داده می‌شه (جزئیات: D13). `save_plan`/`load_plan` فایل `plan.json` رو اتمی و فقط با `version` بزرگ‌تر می‌نویسن و رویداد `plan.saved` ثبت می‌کنن (D14). `save_clarifications`/`load_clarifications` فایل `clarifications.json` (مدل `Clarifications` در `automa.plan`) رو اتمی و فقط یک بار می‌نویسن (رویداد `clarifications.saved`)؛ نبود فایل یعنی فهرست خالی.
فایل‌محور بودن یعنی وضعیت قابل خوندن، قابل ادامه و قابل بررسی دستیه.

### حافظه (مرحله بعد)
سه لایه جدا: ترجیحات کاربر، دانش اختصاصی هر پروژه، و درس‌های عمومی (مثلاً «سرویس X برای Y مناسب نبود»). درس‌ها بعد از پایان هر پروژه توسط یک مرحله «بازبینی» استخراج می‌شن.

### Skill Engine (مرحله بعد)
Skillها با قالب استاندارد `SKILL.md` (frontmatter + دستورالعمل) در `skills/` نگهداری می‌شن. انتخاب Skill بر اساس توضیحش انجام می‌شه. ساخت خودکار Skill آخرین مرحله‌ست و فقط بعد از اینکه مجموعه آزمون ارزیابی قوی شد فعال می‌شه؛ هر Skill جدید قبل از ورود باید روی آزمون‌ها بهتر یا مساوی عمل کنه.

## فناوری‌ها

| بخش | انتخاب | دلیل |
|---|---|---|
| زبان | Python 3.11+ | اکوسیستم AI، سادگی |
| کلاینت مدل | `httpx` با API سازگار OpenAI | بدون وابستگی به SDK یک شرکت |
| ذخیره‌سازی | فایل JSON + SQLite | بدون سرور جدا، قابل خوندن |
| مدل داده | `pydantic` | اعتبارسنجی خروجی‌های JSON مدل‌ها |
| رابط | CLI با `typer` | ساده؛ رابط وب در مراحل بعد |
| تست | `pytest` + FakeProvider | توسعه بدون هزینه API |
| استقرار | Docker / docker compose | انتقال آسان به سرور |

## مجموعه آزمون (Benchmark)

پوشه `benchmarks/` شامل چند هدف نمونه با معیار موفقیت مشخصه. بعد از هر تغییر بزرگ، سیستم روی این‌ها اجرا می‌شه (در حالت واقعی، با بودجه محدود) و نتیجه در `benchmarks/results/` ثبت می‌شه. هیچ تغییری که نتیجه benchmark رو بدتر کنه نباید ادغام بشه.
