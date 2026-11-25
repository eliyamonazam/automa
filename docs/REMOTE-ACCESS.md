# باز کردن Automa از بیرون خونه (Cloudflare Tunnel)

این راهنما برای **صاحب Automa** است (کسی که Automa روی سیستمش نصبه). آخرش خانواده می‌تونن از گوشی، با اینترنت موبایل یا هر جای دیگه، آدرسی مثل `https://automa.example.com` رو باز کنن، وارد حساب خودشون بشن و ایده بدن.

چطوری کار می‌کنه، خیلی کوتاه: Automa همچنان فقط روی خود سیستم تو (`127.0.0.1`) گوش می‌ده و هیچ پورتی روی مودم باز نمی‌شه. برنامهٔ کوچک `cloudflared` از سیستم تو به Cloudflare وصل می‌شه و درخواست‌های اون آدرس رو با HTTPS به Automa می‌رسونه (D187، D188).

> زمان لازم: حدود نیم ساعت، فقط یک بار. همهٔ دستورها در برنامهٔ **Ubuntu** (همون WSL که Automa داخلشه) زده می‌شن، مگر جایی که نوشته «در Windows».

## چیزهایی که خودت باید فراهم کنی

Automa این‌ها رو نمی‌تونه برات بسازه:

1. **یک دامنه** (مثلاً `example.com`) که مال خودته. اگه نداری، باید یکی بخری (از خود Cloudflare یا هر فروشندهٔ دامنه).
2. **یک حساب رایگان Cloudflare** که دامنه‌ت توش اضافه شده و DNSش با Cloudflare است (در داشبورد Cloudflare: «Add a site» و بعد عوض کردن nameserverها پیش فروشندهٔ دامنه، طبق راهنمای خود Cloudflare). تا وضعیت دامنه در Cloudflare «Active» نشده، قدم‌های پایین کار نمی‌کنن.
3. **یک نام برای Automa** روی همون دامنه، مثلاً `automa.example.com`. در همهٔ دستورهای پایین `automa.example.com` رو با نام خودت عوض کن.

## قدم ۱ — حساب‌ها (بدون حساب، حالت بیرونی اصلاً روشن نمی‌شه)

اول برای خودت یک حساب owner بساز، بعد برای هر نفر از خانواده یک حساب member:

```bash
cd ~/automa && . .venv/bin/activate
automa users add parsa --role owner
automa users add baba
automa users list
```

هر دستور رمز رو می‌پرسه. رمز هر نفر رو طولانی و جدا انتخاب کن؛ این آدرس روی اینترنته. بعد از ۵ رمز اشتباه، حساب برای مدتی قفل می‌شه (`automa users passwd <نام>` قفل رو باز می‌کنه). پروژه‌های member همیشه در حالت امن اجرا می‌شن (دستورها فقط داخل Docker، تأییدها با تو) و سقف خرج ماهانهٔ خودشون رو دارن.

## قدم ۲ — نصب cloudflared

این دستورها همون دستورهای راهنمای رسمی Cloudflare برای Ubuntu هستن:

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared
cloudflared --version
```

اگه آخرین دستور یک شمارهٔ نسخه نشون داد، نصب شده.

## قدم ۳ — وصل کردن cloudflared به حساب Cloudflare

```bash
cloudflared tunnel login
```

یک لینک چاپ می‌شه. اون رو در مرورگر Windows باز کن، وارد Cloudflare شو و دامنه‌ت رو انتخاب کن. بعدش در Ubuntu پیام موفقیت میاد و فایل `cert.pem` در پوشهٔ `~/.cloudflared` ذخیره می‌شه.

**این دستورها رو با `sudo` نزن**؛ وگرنه فایل‌ها در پوشهٔ root می‌رن و میان‌بر پیداشون نمی‌کنه.

## قدم ۴ — ساختن tunnel و وصل کردن نام

```bash
cloudflared tunnel create automa
cloudflared tunnel route dns automa automa.example.com
```

دستور اول tunnelی به اسم `automa` می‌سازه و فایل کلیدش رو در `~/.cloudflared` می‌ذاره (این فایل و `cert.pem` مثل رمزن: جایی نفرستشون و هرگز در مخزن نذارشون). دستور دوم در DNS دامنه‌ت نام `automa.example.com` رو به همین tunnel وصل می‌کنه.

فایل `config.yml` لازم نیست: میان‌بر Automa نشونی Automa رو خودش به tunnel می‌ده.

## قدم ۵ — آزمایش با میان‌بر

میان‌بر Automa (بخش Windows در `README.md`) حالا یک گزینهٔ تازه داره: `-PublicHost`.

1. فایل‌های تازهٔ میان‌بر رو یک بار دیگه به Windows کپی کن (همون دستور قدم ۱ بخش Windows در README؛ `<نام‌کاربری>` نام کاربری Windows توئه):

   ```bash
   cp ~/automa/scripts/automa-*.bat ~/automa/scripts/automa-*.ps1 /mnt/c/Users/<نام‌کاربری>/Documents/automa-launcher/
   ```

2. **در Windows**: روی میان‌بر Automa روی دسکتاپ راست‌کلیک ← Properties. در قسمت **Target** بعد از `automa-web.bat` یک فاصله و بعد این رو اضافه کن، و OK بزن:

   ```text
   -PublicHost automa.example.com
   ```

   اگه اسم tunnel رو چیزی غیر از `automa` گذاشتی، `-TunnelName <اسم>` رو هم اضافه کن.

3. روی میان‌بر دابل‌کلیک کن. در پنجرهٔ سیاه باید ببینی:
   `From outside the house: https://automa.example.com/ (only while this window is open).`

4. با گوشی، **وای‌فای خونه رو خاموش کن** (تا مطمئن بشی از بیرون کار می‌کنه) و `https://automa.example.com` رو باز کن. صفحهٔ ورود Automa باید بیاد. با حساب member وارد شو و ببین فقط پروژه‌های همون حساب دیده می‌شه.

پشت صحنه، میان‌بر داخل WSL `scripts/automa-tunnel.sh` رو اجرا می‌کنه: tunnel رو با `cloudflared tunnel run --url http://127.0.0.1:8765 automa` روشن می‌کنه و بعد `automa web --allow-actions --public --host-name automa.example.com` رو. هر وقت پنجره رو ببندی یا Ctrl+C بزنی، هر دو با هم خاموش می‌شن. گزارش tunnel در `~/.cloudflared/automa-tunnel-automa.log` نوشته می‌شه.

بدون میان‌بر هم می‌شه (مثلاً روی Linux): در پوشهٔ Automa بزن
`bash scripts/automa-tunnel.sh automa automa.example.com --port 8765 --allow-actions`.

## قدم ۶ — روشن موندن سیستم و اجرای خودکار

Automa فقط وقتی از بیرون باز می‌شه که سیستم روشن، بیدار و پنجرهٔ میان‌بر باز باشه.

- **خواب نره (در Windows):** Settings ← System ← Power & battery ← «Screen, sleep, & hibernate timeouts» ← خواب (Sleep) وقتی به برق وصله رو **Never** بذار. خاموش شدن صفحه‌نمایش اشکالی نداره. یا در PowerShell:

  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  ```

- **بعد از روشن شدن سیستم خودش باز بشه (در Windows):** کلید Win+R رو بزن، `shell:startup` رو بنویس و Enter بزن. یک کپی از میان‌بر Automa (همونی که `-PublicHost` داره) رو در این پوشه بذار. از این به بعد بعد از هر ورود به Windows پنجرهٔ Automa خودش باز می‌شه. اگه نمی‌خوای مرورگر هم باز بشه، در Target این کپی ` -NoBrowser` رو هم اضافه کن.
- **به‌روزرسانی Windows:** در Settings ← Windows Update ← Advanced options ← «Active hours» ساعت‌هایی که خانواده معمولاً Automa رو استفاده می‌کنن بذار تا وسطش ری‌استارت نشه. بعد از ری‌استارت، اگه وارد Windows بشی، میان‌بر پوشهٔ startup دوباره بازش می‌کنه.

## قدم ۷ — فرستادن به خانواده

به هر نفر فقط این‌ها رو بده: آدرس (`https://automa.example.com`)، نام کاربری و رمز خودش. رمز رو حضوری یا تلفنی بگو، نه در گروه‌های پیام‌رسان. روی گوشی، بعد از باز کردن آدرس در مرورگر، از منوی مرورگر «Add to Home screen» رو بزنن تا مثل یک برنامه روی صفحه بیاد.

## امنیت، خیلی کوتاه

- هر صفحه ورود می‌خواد؛ فرم‌ها فقط از همون آدرس `https://` پذیرفته می‌شن؛ اتصال با HTTPS است (HSTS).
- رمزهای اشتباه هر بازدیدکننده جدا شمرده می‌شن، پس یک غریبه نمی‌تونه با حدس رمز ورود خانواده رو ببنده؛ هر حساب هم بعد از ۵ تلاش اشتباه قفل می‌شه.
- اگه کسی از خانواده دیگه نباید وارد بشه: `automa users disable <نام>`.
- برای بستن کامل دسترسی بیرونی: پنجرهٔ Automa رو ببند، یا `-PublicHost …` رو از Target میان‌بر پاک کن. اگه دیگه هیچ‌وقت لازم نیست، در Ubuntu: `cloudflared tunnel delete automa` و بعد رکورد DNS نام `automa` رو در داشبورد Cloudflare پاک کن.
- اختیاری، برای امنیت بیشتر: Cloudflare Access (Zero Trust) می‌تونه قبل از Automa یک مرحلهٔ ورود با ایمیل هم بذاره. Automa برای کار کردن بهش نیاز نداره.

## وقتی چیزی خراب شد

| چی می‌بینی | یعنی چی | چی کار کنی |
|---|---|---|
| پنجره: `cloudflared is not installed in WSL` | قدم ۲ انجام نشده | قدم ۲ رو انجام بده، یا میان‌بر رو بدون `-PublicHost` باز کن |
| پنجره: `--public needs at least one active account first` | هیچ حساب فعالی نیست | قدم ۱ |
| پنجره: `This Automa copy has no scripts/automa-tunnel.sh yet` | Automa قدیمیه | اول «به‌روزرسانی Automa» رو بزن |
| پنجره: `WARNING: the tunnel stopped` | tunnel وصل نشد (معمولاً اسم tunnel اشتباهه یا قدم ۳/۴ کامل نشده) | فایل گزارش که اسمش در پیام هست رو ببین؛ `cloudflared tunnel list` باید tunnel `automa` رو نشون بده |
| گوشی: صفحهٔ خطای Cloudflare (مثلاً 502 یا 1033) | پنجرهٔ Automa بسته‌ست، سیستم خوابه یا اینترنت خونه قطعه | سیستم رو بیدار کن و میان‌بر رو باز کن |
| گوشی: صفحهٔ Automa با «نشانی صفحه درست نیست» (کد 421) | نام `-PublicHost` با آدرسی که باز شده یکی نیست | Target میان‌بر رو با همون نامی که در قدم ۴ دادی درست کن |
| `apt-get update` خطای `NO_PUBKEY` می‌ده | کلید بستهٔ Cloudflare عوض شده | دستور دوم قدم ۲ (`curl … cloudflare-main.gpg`) رو دوباره بزن |
| خانواده رمز رو اشتباه زد و قفل شد | قفل موقت بعد از ۵ تلاش | `automa users passwd <نام>` |

اگه هیچ‌کدوم نبود، متن پنجرهٔ سیاه و فایل `~/.cloudflared/automa-tunnel-automa.log` رو نگه دار.
