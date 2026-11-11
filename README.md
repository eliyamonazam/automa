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

## راه‌اندازی

```bash
cp .env.example .env      # کلید API و آدرس درگاه رو اینجا بذار
cp automa.toml.example automa.toml
pip install -e .          # یا با Docker (پایین‌تر)
automa new "ایده من ..."   # سؤال‌ها + ساخت plan.json
automa run <project-id>    # اجرا/ادامه: Worker ← Verifier ← اصلاح
automa status             # فهرست پروژه‌ها، هزینه و برنامه
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
