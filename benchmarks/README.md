# مجموعه آزمون (Benchmark)

هر فایل `<id>.toml` در این پوشه یک هدف نمونه با **معیار موفقیت قطعی**ه (قالب: `automa.benchmarks.Benchmark`، تصمیم D26).

| شناسه | نوع | هدف |
|---|---|---|
| `cli-word-count` | `cli` | ابزار CLI شمارش خط/کلمه/کاراکتر در Python |
| `static-landing-page` | `web` | صفحه وب ایستا برای یک کافه خیالی |
| `research-brief` | `research` | گزارش تحقیقی کوتاه SQLite در برابر PostgreSQL |

فیلدها: `id` (برابر نام فایل)، `title`، `kind`، `goal`، `answer` (جواب همه سؤال‌های روشن‌کننده، چون اجرا بدون حضور کاربره)، `budget_usd` (سقف بودجه پروژه، حداکثر ۵)، `max_attempts`، و `success_criteria` که **همه** باید `check` داشته باشن (`file_exists`، `file_contains`، `command`). موفقیت هیچ‌وقت با قضاوت مدل تعیین نمی‌شه.

ارزیابی: `automa.benchmarks.evaluate_benchmark(benchmark, <project>/files, events)` بررسی‌ها رو از Tool Layer با `verifier_policy()` اجرا می‌کنه. اجرا: `automa bench list` و `automa bench run <id>` (با کلید API واقعی هزینه داره؛ سقف پروژه = کمترِ `budget_usd` و سقف `automa.toml`). هر اجرا یک پروژه جدید `bench-<زمان>-<id>` در workspace می‌سازه و نتیجه‌اش رو یک‌بار در `results/<id>/<project-id>.json` می‌نویسه (`passed`، `score`، بررسی‌ها، دلیل توقف، هزینه، تعداد فراخوانی و تلاش‌ها). کد خروج ۰ یعنی همه معیارها قبول شدن، ۸ یعنی ثبت شد ولی قبول نشد (تصمیم D27).
