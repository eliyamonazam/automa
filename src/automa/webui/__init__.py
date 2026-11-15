"""Simple read-only web UI for project status and pending approvals (D77).

This package only builds views and renders HTML; it never decides approvals,
writes files or calls a model.
"""

from automa.webui.render import CONTENT_SECURITY_POLICY, render_overview, render_project, text
from automa.webui.views import (
    ApprovalView,
    Overview,
    ProjectRow,
    ProjectView,
    TaskRow,
    approval_view,
    build_overview,
    build_project_view,
)

__all__ = [
    "CONTENT_SECURITY_POLICY",
    "ApprovalView",
    "Overview",
    "ProjectRow",
    "ProjectView",
    "TaskRow",
    "approval_view",
    "build_overview",
    "build_project_view",
    "render_overview",
    "render_project",
    "text",
]
