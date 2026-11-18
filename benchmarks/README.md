# مجموعه آزمون (Benchmark)

هر فایل `<id>.toml` در این پوشه یک هدف نمونه با **معیار موفقیت قطعی**ه (قالب: `automa.benchmarks.Benchmark`، تصمیم D26).

| شناسه | نوع | هدف |
|---|---|---|
| `cli-word-count` | `cli` | ابزار CLI شمارش خط/کلمه/کاراکتر در Python |
| `static-landing-page` | `web` | صفحه وب ایستا برای یک کافه خیالی |
| `research-brief` | `research` | گزارش تحقیقی کوتاه SQLite در برابر PostgreSQL |
| `landing-page-with-image` | `web` | صفحه فرود فروشگاه بادبادک با یک تصویر ساخته‌شده (`needs_images`) |

فیلدها: `id` (برابر نام فایل)، `title`، `kind`، `goal`، `answer` (جواب همه سؤال‌های روشن‌کننده، چون اجرا بدون حضور کاربره)، `budget_usd` (سقف بودجه پروژه، حداکثر ۵)، `max_attempts`، `needs_images` (اختیاری؛ `true` یعنی هدف به ابزار `generate_image` نیاز داره و `automa bench run` بدون `[images] model` قبل از ساخت پروژه با خطای تنظیمات متوقف می‌شه، D104)، و `success_criteria` که **همه** باید `check` داشته باشن (`file_exists`، `file_contains`، `command`). موفقیت هیچ‌وقت با قضاوت مدل تعیین نمی‌شه.

ارزیابی: `automa.benchmarks.evaluate_benchmark(benchmark, <project>/files, events)` بررسی‌ها رو از Tool Layer با `verifier_policy()` اجرا می‌کنه. اجرا: `automa bench list` و `automa bench run <id>` (با کلید API واقعی هزینه داره؛ سقف پروژه = کمترِ `budget_usd` و سقف `automa.toml`). هر اجرا یک پروژه جدید `bench-<زمان>-<id>` در workspace می‌سازه و نتیجه‌اش رو یک‌بار در `results/<id>/<project-id>.json` می‌نویسه (`passed`، `score`، بررسی‌ها، دلیل توقف، هزینه، تعداد فراخوانی و تلاش‌ها). کد خروج ۰ یعنی همه معیارها قبول شدن، ۸ یعنی ثبت شد ولی قبول نشد (تصمیم D27).

تحلیل: `automa bench report [--benchmark <id>] [--results DIR] [--json]` همه نتیجه‌ها رو می‌خونه (بدون کلید، بدون فراخوانی مدل، بدون نوشتن) و برای هر benchmark نرخ قبولی، میانگین امتیاز و تلاش‌ها، هزینه، دلایل توقف، دسته علت شکست (`criteria_missed` = پروژه کامل شد ولی معیار benchmark رد شد، `tasks_failed`، `planning`، `budget`، `provider`، `other_stop`) و معیارهای ردشده رو نشون می‌ده. فایل‌های خراب یا نامعتبر نادیده گرفته نمی‌شن: به‌عنوان `problem` چاپ می‌شن و کد خروج ۱ می‌شه (تصمیم D28).

benchmark تصویری (`landing-page-with-image`): بررسی‌ها قطعی‌ان: دقیقاً یک فایل `images/hero.png` یا `images/hero.jpg` (نه symlink)، PNG/JPEG واقعی هم‌قالب پسوند با ابعاد حداقل ۲۵۶×۲۵۶ (خواندن سرآیند IHDR/SOF)، یک `<img>` با `src` نسبی همون فایل و `alt` غیرخالی، و بدون JavaScript، URL بیرونی یا `data:`. چون بررسی `command` بدون shell و تک‌خطیه (D92)، هر بررسی یک برنامهٔ `python3 -c "exec('...')"`ه. اینکه تصویر واقعاً با `generate_image` ساخته شده باشه با فایل‌ها قابل اثبات نیست؛ تعداد فراخوانی‌های مدل تصویری در Ledger (و `model_calls` نتیجه) دیده می‌شه (D104).
