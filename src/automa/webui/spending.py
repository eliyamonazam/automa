"""The "My spending this month" page (M15, D184).

``GET /spending`` shows the signed-in account what it spent this month, project by
project, and how much of its monthly cap (D183) is used; an owner account also gets
everyone's spending, one row per account plus the projects that belong to nobody
(made from the command line or before accounts existed). Without accounts the page
shows the whole workspace, as the top bar already does.

Two different month figures are shown on purpose, both exact Ledger sums:

* "this month" is the local calendar month (``[display] timezone``, D139), the same
  number as the top bar, attributed to an account through its projects' ``owner``;
* "counted toward the cap" is exactly what ``BudgetGuard`` compares against the
  monthly cap: the rows recorded with that account's ``user`` in the current UTC
  month (D183). Rows recorded before D183 have no ``user`` and are not counted.

Display only: nothing here reads or changes a limit in a way the guard sees, and the
Ledger is opened read-only by the server. Pure builders and renderers, no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, tzinfo
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict

from automa.budget import NANO_PER_USD, LedgerEntry
from automa.budget.ledger import month_of
from automa.l10n import LANGUAGES, Language, localize_digits
from automa.webui.i18n import check_messages, template_fields
from automa.webui.money import nusd_money
from automa.webui.render import ICONS, page, text

SPENDING_PATH = "/spending"
MAX_PROJECTS_SHOWN = 10

Kind = Literal["owner", "member", "unowned", "gone"]

_RAW: dict[str, dict[Language, str]] = {
    "title": {"en": "My spending this month", "fa": "خرج من این ماه"},
    "title.everyone": {"en": "Spending this month", "fa": "خرج این ماه"},
    "lead": {
        "en": "What your projects cost this month, and how much of your monthly limit is used.",
        "fa": "پروژه‌هات این ماه چقدر خرج کردن و چقدر از سقف ماهانه‌ت مونده.",
    },
    "lead.plain": {
        "en": "What your projects cost this month.",
        "fa": "پروژه‌هات این ماه چقدر خرج کردن.",
    },
    "lead.workspace": {
        "en": "What all projects cost this month.",
        "fa": "همهٔ پروژه‌ها این ماه چقدر خرج کردن.",
    },
    "month": {"en": "This month", "fa": "این ماه"},
    "today": {"en": "Today: {amount}", "fa": "امروز: {amount}"},
    "cap.heading": {"en": "Monthly limit", "fa": "سقف ماهانه"},
    "cap.limit": {"en": "Limit: {amount}", "fa": "سقف: {amount}"},
    "cap.used": {"en": "Counted so far: {amount}", "fa": "تا الان شمرده‌شده: {amount}"},
    "cap.left": {"en": "Left: {amount}", "fa": "مونده: {amount}"},
    "cap.full": {
        "en": "This month's limit is used up: new work stops until next month, "
        "or until the owner of this Automa raises your limit.",
        "fa": "سقف این ماه پر شده: کار تازه تا ماه بعد متوقف می‌مونه، "
        "یا تا وقتی صاحب این Automa سقفت رو بیشتر کنه.",
    },
    "cap.near": {
        "en": "Most of this month's limit is used.",
        "fa": "بیشتر سقف این ماه خرج شده.",
    },
    "cap.none": {
        "en": "You have no monthly limit; each project still has its own limit.",
        "fa": "سقف ماهانه نداری؛ هر پروژه همچنان سقف خودش رو داره.",
    },
    "cap.utc": {
        "en": "The limit counts from the first day of the month in UTC, so near the "
        "start of a month it can differ a little from “this month” above.",
        "fa": "سقف از اول ماه میلادی به وقت جهانی (UTC) شمرده می‌شه؛ برای همین نزدیک "
        "اول ماه ممکنه کمی با «این ماه» بالا فرق کنه.",
    },
    "project_cap": {
        "en": "Limit of each project: {cap}",
        "fa": "سقف هر پروژه: {cap}",
    },
    "projects.heading": {"en": "Projects this month", "fa": "پروژه‌های این ماه"},
    "projects.none": {
        "en": "Nothing spent this month yet.",
        "fa": "این ماه هنوز خرجی نشده.",
    },
    "projects.more": {
        "en": "and {count} more projects",
        "fa": "و {count} پروژهٔ دیگه",
    },
    "project.gone": {"en": "(deleted project)", "fa": "(پروژهٔ پاک‌شده)"},
    "project.total": {"en": "all time {amount}", "fa": "از اول {amount}"},
    "everyone.heading": {"en": "Everyone", "fa": "همه"},
    "everyone.lead": {
        "en": "Only you see this part (owner account).",
        "fa": "این بخش رو فقط تو می‌بینی (حساب صاحب).",
    },
    "everyone.total": {"en": "Total this month: {amount}", "fa": "جمع این ماه: {amount}"},
    "kind.owner": {"en": "owner", "fa": "صاحب"},
    "kind.member": {"en": "member", "fa": "عضو"},
    "kind.unowned": {
        "en": "No account (command line or older projects)",
        "fa": "بدون حساب (خط فرمان یا پروژه‌های قدیمی)",
    },
    "kind.gone": {"en": "account removed", "fa": "حساب حذف‌شده"},
    "disabled": {"en": "disabled", "fa": "غیرفعال"},
    "row.projects": {"en": "{count} projects this month", "fa": "{count} پروژه این ماه"},
    "row.no_cap": {"en": "no monthly limit", "fa": "بدون سقف ماهانه"},
    "row.cap": {"en": "{used} of {cap} limit", "fa": "{used} از سقف {cap}"},
    "unreadable": {
        "en": "The spending record could not be read; nothing is shown.",
        "fa": "دفتر خرج خونده نشد؛ چیزی نشون داده نمی‌شه.",
    },
    "back": {"en": "← all projects", "fa": "→ همهٔ پروژه‌ها"},
}

check_messages(_RAW)
MESSAGES: Mapping[str, Mapping[Language, str]] = MappingProxyType(
    {k: MappingProxyType(dict(v)) for k, v in _RAW.items()}
)


def st(key: str, lang: Language, **params: object) -> str:
    """Like :func:`automa.webui.i18n.t` for this module's table."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang!r}")
    template = MESSAGES[key][lang]
    if set(params) != template_fields(template):
        raise ValueError(f"{key}: parameters {sorted(params)}")
    filled: dict[str, str] = {}
    for k, v in params.items():
        if isinstance(v, int) and not isinstance(v, bool):
            filled[k] = localize_digits(str(v), lang)
        elif isinstance(v, str):
            filled[k] = v
        else:
            raise TypeError("message parameters are str or int")
    return template.format(**filled)


# -- view models --------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectSpend(_Model):
    id: str
    title: str | None  # None: the project is gone (deleted or unreadable)
    month_nusd: int
    total_nusd: int


class PersonSpend(_Model):
    """One account (or the "no account" bucket) this month."""

    name: str | None
    kind: Kind
    disabled: bool = False
    month_nusd: int = 0
    today_nusd: int = 0
    # The monthly cap the guard enforces (D183) and what it counts so far.
    cap_nusd: int | None = None
    cap_used_nusd: int = 0
    project_cap_nusd: int | None = None
    caps_known: bool = True
    projects: tuple[ProjectSpend, ...] = ()

    @property
    def left_nusd(self) -> int | None:
        return None if self.cap_nusd is None else max(self.cap_nusd - self.cap_used_nusd, 0)

    @property
    def cap_percent(self) -> int | None:
        if self.cap_nusd is None or self.cap_nusd <= 0:
            return None
        return min(100, self.cap_used_nusd * 100 // self.cap_nusd)


class SpendingView(_Model):
    me: PersonSpend
    everyone: tuple[PersonSpend, ...] | None = None  # owner accounts only
    error: str | None = None  # the Ledger could not be read


class ProjectInfo(_Model):
    """What the page needs of a project: its owner and a title."""

    owner: str | None
    title: str


class AccountInfo(_Model):
    """An account of ``users.json``, its role and the caps the guard applies to it."""

    name: str
    role: Literal["owner", "member"]
    disabled: bool = False
    cap_usd: float | None = None
    project_cap_usd: float | None = None
    caps_known: bool = True  # False: the caps are not known here, so none is claimed


def _nusd(usd: float | None) -> int | None:
    if usd is None:
        return None
    return int((Decimal(str(usd)) * NANO_PER_USD).to_integral_value())


def attribute(entry: LedgerEntry, projects: Mapping[str, ProjectInfo]) -> str | None:
    """Whose spending a Ledger row is: the owner of its project; for a project that no
    longer exists, the account recorded on the row (D183); otherwise nobody's."""
    if entry.project_id is not None and entry.project_id in projects:
        return projects[entry.project_id].owner
    return entry.user


def _person(
    name: str | None,
    kind: Kind,
    rows: Iterable[LedgerEntry],
    all_rows: Iterable[LedgerEntry],
    projects: Mapping[str, ProjectInfo],
    *,
    now: datetime,
    tz: tzinfo,
    cap_usd: float | None = None,
    project_cap_usd: float | None = None,
    disabled: bool = False,
    caps_known: bool = True,
) -> PersonSpend:
    local_now = now.astimezone(tz)
    utc_month = month_of(now.astimezone(UTC).date())
    month = today = 0
    per_month: dict[str, int] = {}
    for e in rows:
        local = e.ts.astimezone(tz)
        if (local.year, local.month) != (local_now.year, local_now.month):
            continue
        month += e.cost_nusd
        if local.date() == local_now.date():
            today += e.cost_nusd
        if e.project_id is not None:
            per_month[e.project_id] = per_month.get(e.project_id, 0) + e.cost_nusd
    totals: dict[str, int] = {}
    cap_used = 0
    for e in all_rows:
        if e.project_id in per_month:
            totals[e.project_id] = totals.get(e.project_id, 0) + e.cost_nusd
        # Exactly the guard's figure (D183): this account's rows in this UTC month.
        if (
            name is not None
            and e.user == name
            and month_of(e.ts.astimezone(UTC).date()) == (utc_month)
        ):
            cap_used += e.cost_nusd
    spends = [
        ProjectSpend(
            id=pid,
            title=projects[pid].title if pid in projects else None,
            month_nusd=spent,
            total_nusd=totals.get(pid, spent),
        )
        for pid, spent in per_month.items()
    ]
    spends.sort(key=lambda p: (-p.month_nusd, p.id))
    return PersonSpend(
        name=name,
        kind=kind,
        disabled=disabled,
        month_nusd=month,
        today_nusd=today,
        cap_nusd=_nusd(cap_usd),
        cap_used_nusd=cap_used,
        project_cap_nusd=_nusd(project_cap_usd),
        caps_known=caps_known,
        projects=tuple(spends),
    )


def build_spending(
    entries: list[LedgerEntry],
    projects: Mapping[str, ProjectInfo],
    *,
    viewer: AccountInfo | None,
    accounts: Iterable[AccountInfo] = (),
    workspace_project_cap_usd: float | None = None,
    now: datetime,
    tz: tzinfo,
    error: str | None = None,
) -> SpendingView:
    """The page for ``viewer`` (``None``: no accounts, the whole workspace).

    A member gets only their own figures; an owner also gets ``everyone``: each
    account of ``accounts``, any other name that projects still carry (an account
    since removed) and the "no account" bucket, each only when it has a row or is an
    account."""
    if error is not None:
        entries = []
    if viewer is None:
        me = _person(
            None,
            "unowned",
            entries,
            entries,
            projects,
            now=now,
            tz=tz,
            project_cap_usd=workspace_project_cap_usd,
        )
        return SpendingView(me=me, error=error)
    by_owner: dict[str | None, list[LedgerEntry]] = {}
    for e in entries:
        by_owner.setdefault(attribute(e, projects), []).append(e)

    def person(account: AccountInfo) -> PersonSpend:
        return _person(
            account.name,
            account.role,
            by_owner.get(account.name, []),
            entries,
            projects,
            now=now,
            tz=tz,
            cap_usd=account.cap_usd,
            project_cap_usd=account.project_cap_usd,
            disabled=account.disabled,
            caps_known=account.caps_known,
        )

    me = person(viewer)
    if viewer.role != "owner":
        return SpendingView(me=me, error=error)
    known = sorted(accounts, key=lambda a: (a.role != "owner", a.name))
    rows = [person(a) for a in known]
    names = {a.name for a in known}
    for name in sorted(n for n in by_owner if n is not None and n not in names):
        rows.append(_person(name, "gone", by_owner[name], entries, projects, now=now, tz=tz))
    unowned = _person(None, "unowned", by_owner.get(None, []), entries, projects, now=now, tz=tz)
    if unowned.month_nusd > 0:
        rows.append(unowned)
    return SpendingView(me=me, everyone=tuple(rows), error=error)


# -- rendering ----------------------------------------------------------------------------


def render_spending(view: SpendingView, *, rate: float | None, lang: Language) -> str:
    """The whole page: the viewer's month, cap and projects, then (owner) everyone."""

    def m(nusd: int) -> str:
        return text(nusd_money(nusd, rate, lang=lang))

    me = view.me
    heading = st("title" if me.name is not None else "title.everyone", lang)
    if me.name is None:
        lead = st("lead.workspace", lang)
    else:
        lead = st("lead" if me.caps_known else "lead.plain", lang)
    parts = [
        f'<p class="muted"><a href="/">{text(st("back", lang))}</a></p>',
        f'<header class="home-head"><h1>{text(heading)}</h1>'
        f'<p class="muted page-lead">{text(lead)}</p></header>',
    ]
    if view.error is not None:
        parts.append(
            f'<section class="card err"><p>{text(st("unreadable", lang))}</p>'
            f'<p class="muted" dir="auto">{text(view.error)}</p></section>'
        )
        return page(heading, "".join(parts), lang=lang)
    today = nusd_money(me.today_nusd, rate, lang=lang)
    parts.append(
        '<section class="card spend-total">'
        f'<p class="spend">{ICONS["wallet"]}<span>{text(st("month", lang))} '
        f"<strong>{m(me.month_nusd)}</strong></span>"
        f'<span class="muted">{text(st("today", lang, amount=today))}</span></p>'
        f"{_cap(me, lang, m)}</section>"
    )
    parts.append(_projects(me, lang, m))
    if view.everyone is not None:
        total = sum(p.month_nusd for p in view.everyone)
        rows = "".join(_row(p, lang, m) for p in view.everyone)
        parts.append(
            f"<h2>{text(st('everyone.heading', lang))}</h2>"
            f'<p class="muted">{text(st("everyone.lead", lang))} · '
            f"{text(st('everyone.total', lang, amount=nusd_money(total, rate, lang=lang)))}</p>"
            f'<div class="cards people">{rows}</div>'
        )
    return page(heading, "".join(parts), lang=lang)


def _with_amount(key: str, lang: Language, amount_html: str) -> str:
    """The escaped message ``key`` with its ``{amount}`` field set to ready HTML."""
    return text(st(key, lang, amount="{amount}")).replace("{amount}", amount_html)


def _bar(percent: int, tone: str) -> str:
    return (
        f'<div class="bar {tone}" role="progressbar" aria-valuenow="{percent}" '
        f'aria-valuemin="0" aria-valuemax="100"><div class="fill" style="width:{percent}%">'
        "</div></div>"
    )


def _tone(p: PersonSpend) -> str:
    percent = p.cap_percent
    if percent is None:
        return ""
    if p.left_nusd == 0:
        return "full"
    return "near" if percent >= 80 else ""


def _cap(p: PersonSpend, lang: Language, m: Callable[[int], str]) -> str:
    out: list[str] = []
    if p.name is not None and p.caps_known:
        out.append(f"<h2>{text(st('cap.heading', lang))}</h2>")
        if p.cap_nusd is None:
            out.append(f'<p class="muted">{text(st("cap.none", lang))}</p>')
        else:
            figures = (
                ("cap.limit", p.cap_nusd),
                ("cap.used", p.cap_used_nusd),
                ("cap.left", p.left_nusd or 0),
            )
            items = "".join(
                f"<li>{_with_amount(key, lang, f'<strong>{m(nusd)}</strong>')}</li>"
                for key, nusd in figures
            )
            tone = _tone(p)
            out.append(_bar(p.cap_percent or 0, tone))
            out.append(f'<ul class="cap-figures">{items}</ul>')
            if tone == "full":
                out.append(f'<p class="cap-note bad">{text(st("cap.full", lang))}</p>')
            elif tone == "near":
                out.append(f'<p class="cap-note">{text(st("cap.near", lang))}</p>')
            out.append(f'<p class="hint">{text(st("cap.utc", lang))}</p>')
    if p.project_cap_nusd is not None:
        cap = st("project_cap", lang, cap="{cap}")
        out.append(f'<p class="meta">{text(cap).replace("{cap}", m(p.project_cap_nusd))}</p>')
    return "".join(out)


def _projects(p: PersonSpend, lang: Language, m: Callable[[int], str]) -> str:
    head = f"<h2>{text(st('projects.heading', lang))}</h2>"
    if not p.projects:
        return f'{head}<p class="muted">{text(st("projects.none", lang))}</p>'
    items = []
    for sp in p.projects[:MAX_PROJECTS_SHOWN]:
        if sp.title is None:
            name = f'<span class="muted">{text(st("project.gone", lang))}</span>'
        else:
            name = f'<a href="/projects/{text(sp.id)}" dir="auto">{text(sp.title)}</a>'
        total = st("project.total", lang, amount="{amount}")
        total_html = text(total).replace("{amount}", m(sp.total_nusd))
        items.append(
            f'<li class="file-row spend-row"><span class="name">{name}</span>'
            f"<strong>{m(sp.month_nusd)}</strong>"
            f'<span class="muted">{total_html}</span></li>'
        )
    more = len(p.projects) - MAX_PROJECTS_SHOWN
    if more > 0:
        items.append(f'<li class="muted">{text(st("projects.more", lang, count=more))}</li>')
    return f'{head}<ul class="spend-list">{"".join(items)}</ul>'


def _row(p: PersonSpend, lang: Language, m: Callable[[int], str]) -> str:
    if p.kind == "unowned":
        title = f"<h3>{text(st('kind.unowned', lang))}</h3>"
    else:
        chips = [text(st(f"kind.{p.kind}", lang))]
        if p.disabled:
            chips.append(text(st("disabled", lang)))
        title = (
            f'<h3><bdi dir="ltr">{text(p.name)}</bdi> '
            f'<span class="chip">{" · ".join(chips)}</span></h3>'
        )
    body = [
        title,
        f'<p class="spend"><strong>{m(p.month_nusd)}</strong>'
        f'<span class="muted">{text(st("row.projects", lang, count=len(p.projects)))}</span></p>',
    ]
    if p.kind in ("owner", "member") and p.caps_known:
        if p.cap_nusd is None:
            body.append(f'<p class="meta">{text(st("row.no_cap", lang))}</p>')
        else:
            cap = text(st("row.cap", lang, used="{used}", cap="{cap}"))
            cap = cap.replace("{used}", m(p.cap_used_nusd)).replace("{cap}", m(p.cap_nusd))
            body.append(_bar(p.cap_percent or 0, _tone(p)))
            body.append(f'<p class="meta">{cap}</p>')
    return f'<section class="card person">{"".join(body)}</section>'
