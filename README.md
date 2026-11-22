# Automa

سیستمی که یک ایده یا هدف رو می‌گیره و با تیمی از Agentهای هوش مصنوعی، تا جای ممکن خودش اون رو به یک خروجی واقعی و قابل استفاده تبدیل می‌کنه.

- **ابزار شخصی:** کاربر اصلی صاحب مخزنه. قرار نیست فروخته بشه.
- **مستقل از ارائه‌دهنده:** هر درگاه سازگار با OpenAI API (مثل OpenRouter یا درگاه‌های مشابه) قابل استفاده‌ست.
- **کنترل مالی سخت‌گیرانه:** سقف بودجه در خود کد اعمال می‌شه، نه فقط با توصیه به یک Agent.
- **معیار اتمام کار، کیفیت واقعیه:** هر خروجی را یک ارزیاب مستقل بررسی می‌کنه.

## اسناد

| سند | محتوا |
|---|---|
| [docs/VISION.md](docs/VISION.md) | سند اصلی ایده (نوشته صاحب پروژه) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | معماری فنی |
| [ROADMAP.md](ROADMAP.md) | نقشه راه و وضعیت هر مرحله |
| [docs/DECISIONS.md](docs/DECISIONS.md) | تصمیم‌های فنی و دلیلشون |
| [CHANGELOG.md](CHANGELOG.md) | گزارش تغییرات هر اجرا |
| [CLAUDE.md](CLAUDE.md) | دستورالعمل اجراهای خودکار توسعه |

## توسعه

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

## Windows

Automa فقط روی Linux و macOS اجرا می‌شه. لایه‌های ایمنی‌اش (محدود کردن دسترسی فایل به پوشه پروژه و کنترل پروسه‌ها) به امکاناتی از POSIX وابسته‌ان که در Windows نیستن، و Automa حاضر نیست با محافظت ضعیف‌تر اجرا بشه. روی Windows از WSL استفاده کن:

```powershell
wsl --install          # در PowerShell با دسترسی Administrator؛ بعد سیستم رو ری‌استارت کن
```

بعد برنامه «Ubuntu» رو از منوی Start باز کن:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone https://github.com/eliyamonazam/automa && cd automa
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
cp .env.example .env   # کلید رو اینجا بذار
automa version
```

### اجرای رابط وب با دابل‌کلیک (بدون ترمینال)

پوشه `scripts` مخزن دو فایل برای Windows داره: `automa-web.bat` و `automa-web.ps1`. فایل `.bat` همون اسکریپت PowerShell رو اجرا می‌کنه، که داخل WSL از پوشه `~/automa` و `.venv` همون‌جا `automa web --allow-actions` رو بالا میاره، صبر می‌کنه تا صفحه جواب بده و `http://127.0.0.1:8765/` رو در مرورگر پیش‌فرض باز می‌کنه. پنجره سیاه باید باز بمونه؛ با Ctrl+C یا بستن پنجره سرور بسته می‌شه (اجرای پس‌زمینه‌ای که از صفحه شروع شده تمیز متوقف/paused می‌شه و بعداً ادامه می‌دی).

۱. یک بار، در Ubuntu، فایل‌ها رو به Windows کپی کن (مثلاً به پوشه Documents؛ `<نام‌کاربری>` رو با نام کاربری Windows عوض کن):

```bash
mkdir -p /mnt/c/Users/<نام‌کاربری>/Documents/automa-launcher
cp ~/automa/scripts/automa-*.bat ~/automa/scripts/automa-*.ps1 /mnt/c/Users/<نام‌کاربری>/Documents/automa-launcher/
```

۲. ساخت میانبر دسکتاپ: در File Explorer روی `automa-web.bat` راست‌کلیک ← «Show more options» ← «Send to» ← «Desktop (create shortcut)». اگه خواستی، در Properties میانبر اسمش رو «Automa» بذار.

۳. از این به بعد فقط روی میانبر دابل‌کلیک کن.

گزینه‌ها (در خط فرمان یا در قسمت Target میانبر بعد از نام فایل): `-ReadOnly` (صفحه فقط‌خواندنی، بدون دکمه‌های تأیید/شروع/توقف)، `-Port 8766` (درگاه دیگر)، `-RepoDir automa2` (کلون در پوشه دیگری از خانه WSL)، `-Distro Ubuntu-24.04` (توزیع غیرپیش‌فرض)، `-NoBrowser`.

نکته‌ها: `-ExecutionPolicy Bypass` در فایل `.bat` فقط برای همون یک پردازه PowerShell اعمال می‌شه و تنظیم سیستم رو عوض نمی‌کنه. اگه پیام «No Automa install found» دیدی، اول مراحل بالا (clone و venv در `~/automa`) رو انجام بده. اگه پنجره می‌گه سرور بالا اومده ولی مرورگر صفحه رو باز نمی‌کنه، احتمالاً forwarding درگاه WSL خاموشه: در فایل `%UserProfile%\.wslconfig` زیر `[wsl2]` خط `networkingMode=mirrored` (یا `localhostForwarding=true`) رو بذار و در PowerShell `wsl --shutdown` بزن. اگه درگاه از قبل اشغال باشه (مثلاً پنجره دیگه‌ای از Automa باز باشه) اسکریپت فقط صفحه رو باز می‌کنه.

### به‌روزرسانی Automa (دابل‌کلیک)

نسخهٔ تازه رو این‌طوری بگیر؛ هیچ‌وقت چیزی از تو پاک یا بازنویسی نمی‌شه:

۱. فقط بار اول (یا اگه فایل‌های پوشهٔ `automa-launcher` قدیمی‌ان)، دستور کپی قدم ۱ بالا رو در Ubuntu دوباره بزن تا `automa-update.bat` و `automa-update.ps1` هم کنار بقیه بیان. اگه هنوز `automa update` رو نداری، یک بار در Ubuntu بزن: `cd ~/automa && git pull --ff-only`.

۲. روی `automa-update.bat` راست‌کلیک ← «Show more options» ← «Send to» ← «Desktop (create shortcut)» و اسم میانبر رو «به‌روزرسانی Automa» بذار.

۳. هر وقت خواستی، روی همین میانبر دابل‌کلیک کن. پنجره نتیجه رو با رنگ نشون می‌ده (سبز: به‌روز شد یا از قبل به‌روز بود، با فهرست تغییرها؛ زرد: چیزی عوض نشد و دلیلش) و آخرش می‌پرسه «Open Automa now?»؛ با Enter خود Automa باز می‌شه.

پشت صحنه همون دستور `automa update` داخل WSL اجرا می‌شه (در Ubuntu هم می‌تونی مستقیم بزنیش): فقط شاخهٔ `main` رو با `git` جلو می‌بره (fast-forward)؛ اگه فایلی رو خودت تغییر داده باشی، روی شاخهٔ دیگه‌ای باشی یا commit محلی داشته باشی، هیچ کاری نمی‌کنه و دقیق می‌گه چی سر راهه و چی کار کنی. `pip install -e .` فقط وقتی دوباره اجرا می‌شه که وابستگی‌ها عوض شده باشن (`--reinstall` برای اجرای اجباری). اگه پنجرهٔ Automa باز بود، بعد از به‌روزرسانی ببندش و دوباره بازش کن.

میانبر خود Automa هم می‌تونه اول به‌روز کنه: در Properties میانبر `automa-web.bat`، قسمت Target، بعد از نام فایل ` -Update` رو اضافه کن؛ اگه به‌روزرسانی نشد، همون نسخهٔ فعلی باز می‌شه.

## راه‌اندازی

```bash
cp .env.example .env      # کلید API و آدرس درگاه رو اینجا بذار
cp automa.toml.example automa.toml
pip install -e .          # یا با Docker (پایین‌تر)
automa new "ایده من ..."   # سؤال‌ها + ساخت plan.json + خط cost range: (بازه هزینه قبل از هر خرجی)
automa run <project-id>    # اجرا/ادامه: Worker ← Verifier ← اصلاح (اگه سر بالای بازه هزینه از [limits] confirm_above_usd، پیش‌فرض ۰٫۳ دلار، بیشتر باشه اول می‌پرسه «ادامه بدم؟»؛ --yes بدون پرسش؛ قبل از اولین فراخوانی مدل خط مشورتی estimate: با تخمین هزینه در برابر بودجه؛ بعد از اجرا خط مشورتی unusual spending: برای فراخوانی‌ها/تلاش‌های غیرعادی همان اجرا)
automa deliver <project-id>     # گزارش تحویل: محصول آماده‌ست؟ فایل‌های اصلی، روش استفاده، چه چیزهایی بررسی شد (--lang fa|en)
automa requests <project-id>    # سؤال‌هایی که Automa وسط کار ازت پرسیده (--all برای جواب‌داده‌ها)
automa answer <project-id> <req> --text "..." [--file PATH]   # جواب بده (متن و/یا فایل)؛ بعد automa run
automa decline <project-id> <req> [--reason "..."]           # «نمی‌تونم بدم»: کار بدون اون ادامه پیدا می‌کنه
automa retry <project-id>       # پروژهٔ ناموفق: مرحله‌های ناموفق با قوی‌ترین مدل دوباره تلاش می‌کنن (--attempts N)؛ بعد automa run
automa run <project-id> --parallel 2   # وظایف مستقل هم‌زمان (یا [orchestrator] max_parallel در automa.toml)
automa run <project-id> --router       # سطح شروع هر وظیفه تازه با Router داده‌محور از لاگ کیفیت (یا [router] enabled؛ --no-router)
automa status             # فهرست پروژه‌ها، هزینه و برنامه
automa workflow save <id>  # ذخیره برنامه پروژه تموم‌شده (همه وظایف قبول) به‌عنوان قالب Workflow (--json)
automa workflows           # فهرست قالب‌های Workflow (فقط‌خوندنی؛ --json)
automa skills              # فهرست Skillها (داخلی و workspace/skills/) و نقش‌هایی که استفاده می‌کنن (فقط‌خوندنی؛ --json؛ اتصال به نقش داخلی: [roles.writer] skills = ["house-style"] در automa.toml)
automa memory list         # حافظه سه‌لایه (کاربر، پروژه، درس‌ها) با متن کامل؛ فقط‌خوندنی (--scope user|project|lesson، --project <id>، --json)
automa memory add "<متن>"  # افزودن یک مورد نوشته‌شده توسط کاربر (--scope، --project؛ source همیشه user؛ بدون کلید و مدل)
automa memory forget <id>  # حذف یک مورد بعد از نمایش و تأیید (--yes برای رد کردن پرسش)
# automa new/run حافظهٔ مرتبط را به‌عنوان داده (نه دستور) به پیام سیستمی اضافه می‌کنند؛
# خاموش کردن: --no-memory یا [memory] in_prompts = false در automa.toml. bench run حافظه نمی‌خواند.
automa review run <id>     # بازبینی پروژه done/failed با مدل (کلید و بودجه؛ هزینه به حساب پروژه) و ذخیره پیشنهادهای حافظه — هنوز حافظه نیستند (--again، --json)
automa review show <id>    # پیشنهادهای آخرین بازبینی با شماره و وضعیت تصمیم (فقط‌خوندنی؛ --review <rid>، --json)
automa review accept <id> <n>…  # نمایش متن کامل، تأیید و نوشتن پیشنهادها در حافظه (source=review؛ --review <rid>، --yes)
automa review reject <id> <n>…  # ثبت رد پیشنهادها بدون نوشتن حافظه (تصمیم نهایی است)
automa quality [--role R]  # نرخ قبولی Verifier و هزینه هر (نقش، مدل) از لاگ کیفیت (فقط‌خوندنی؛ --json؛ بدون کلید)
automa quality sync [<id>]  # وارد کردن تلاش‌های داوری‌شده پروژه‌های قبلی به لاگ کیفیت (idempotent؛ --json)
automa quality route [--role R] [--tier T]  # پیش‌نمایش فقط‌خواندنی: سطحی که Router برای هر نقش انتخاب می‌کند، دلیل و شواهد هر سطح (حتی وقتی Router خاموش است؛ --json)
automa estimate <id> [--max-attempts N] [--router]  # تخمین فقط‌خواندنی هزینهٔ وظایف باقی‌مانده: «حدود X تا Y دلار» (با [display] toman_per_usd = 323000 معادل تومانی هم)، سقف Guard (بدترین حالت) و در برابر بودجهٔ باقی‌ماندهٔ پروژه (--json)
automa finance report [--project ID] [--since D] [--until D]  # خرج واقعی فقط‌خواندنی از Ledger به تفکیک پروژه/روز UTC/مدل/نقش + آخرین تخمین هر پروژه در برابر خرج بعد از آن (--json)
automa new "<هدف>" --workflow <wf-id>  # پروژه جدید با برنامه قالب (بدون Executive، بدون کلید و هزینه)
automa workflow create --goal "<هدف>" --plan plan.json  # قالب جدید از برنامه‌ای که خودتون نوشتید (--json)
automa workflow show <wf-id> [--json | --plan-json]      # نمایش قالب؛ --plan-json فایل قابل ویرایش برنامه
automa workflow edit <wf-id> [--goal "…"] [--plan plan.json]  # ویرایش = قالب جدید با والد؛ قالب قبلی دست نمی‌خوره
automa approvals [<id>]   # درخواست‌های تأیید در انتظار با آرگومان‌های دقیق (فقط‌خوندنی؛ --all، --json)
automa approve <id> <apr-id> [--note ...]  # تأیید یک‌باره (بعد از نمایش و پرسیدن؛ --yes)
automa reject <id> <apr-id> [--note ...]   # رد؛ یادداشت به مدل می‌رسه
automa bench list         # فهرست benchmarkها
automa bench run <id>     # اجرای یک benchmark و ثبت نتیجه در benchmarks/results/
automa bench report       # تحلیل نتیجه‌های ثبت‌شده (فقط‌خوندنی؛ --json، --benchmark <id>)
```

### با Docker

```bash
cp .env.example .env && cp automa.toml.example automa.toml && mkdir -p workspace
docker compose build
docker compose run --rm automa new "ایده من ..."
docker compose run --rm automa run <project-id>
docker compose run --rm automa status
```

کانتینر با کاربر غیر root (پیش‌فرض `1000:1000`؛ با `AUTOMA_UID`/`AUTOMA_GID` عوضش کن تا فایل‌های `workspace/` مال خودت باشن)، فایل‌سیستم فقط‌خوندنی، بدون capability و بدون socket داکر اجرا می‌شه. کلید فقط از `.env` (با `env_file`) میاد و داخل ایمیج نیست. داخل کانتینر `[sandbox] mode` رو `local` نگه دار: `run_command` داخل همین کانتینر ایزوله اجرا می‌شه؛ حالت `docker` اونجا عمداً شکست می‌خوره (D34).

> کلید API هیچ‌وقت نباید وارد مخزن بشه. `.env` در `.gitignore` هست.
