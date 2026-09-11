"""``present_draft`` and the outbound attachment lint.

Regression pins for the 2026-09-08 failure: a seven-attachment client handover
presented with all seven filenames backticked in the body and zero files
attached. The reviewer could read the names and open nothing, so the draft
could not be checked. The standing rule had been followed exactly — which is
why the count now lives in the tool instead of in the model's memory.
"""

import json

import pytest

from tools.present_draft import (
    filenames_mentioned_in_text,
    lint_outbound_reply,
    load_draft,
    present_draft,
)


@pytest.fixture
def pdfs(tmp_path):
    """Two real, non-empty files standing in for review deliverables."""
    made = []
    for name in ("4063_workproduct_contents_EN.pdf", "4063_RFQ_template_JP.pdf"):
        p = tmp_path / name
        p.write_bytes(b"%PDF-1.4\n" + b"x" * 2048)
        made.append(str(p))
    return made


def _call(**kw):
    return json.loads(present_draft(**kw))


# --- rendering the attachment lines ---------------------------------------


def test_every_attachment_gets_a_rendered_media_line(pdfs):
    """The whole point: each file is an openable object, not a name."""
    result = _call(
        to="hk@kotoholdings.com",
        subject="Chita 4063 — handover",
        body="The survey and consultation work product is attached.",
        attachments=pdfs,
    )
    assert result["success"] is True
    rendered = result["rendered"]
    assert result["attachment_count"] == 2
    assert rendered.count("MEDIA:") == 2
    for path in pdfs:
        assert f"MEDIA:{path}" in rendered


def test_attachment_lines_are_numbered_out_of_the_total(pdfs):
    """A count the reviewer can check at a glance."""
    rendered = _call(
        to="a@b.com", subject="s", body="see attached", attachments=pdfs
    )["rendered"]
    assert "**Attachment 1/2:**" in rendered
    assert "**Attachment 2/2:**" in rendered


def test_media_line_count_always_equals_attachment_count(pdfs):
    """The invariant the model used to be asked to maintain by hand."""
    for n in (1, 2):
        result = _call(
            to="a@b.com", subject="s", body="see attached", attachments=pdfs[:n]
        )
        assert result["rendered"].count("MEDIA:") == result["attachment_count"] == n


def test_headers_and_cc_are_rendered_in_full(pdfs):
    rendered = _call(
        to="hk@kotoholdings.com",
        cc="ayano.yamanaka@cleanenergyjapan.jp,atsushi.takeda@cleanenergyjapan.jp",
        subject="Chita 4063",
        body="see attached",
        attachments=pdfs[:1],
    )["rendered"]
    assert "**To:** hk@kotoholdings.com" in rendered
    assert "ayano.yamanaka@cleanenergyjapan.jp" in rendered
    assert "atsushi.takeda@cleanenergyjapan.jp" in rendered


# --- refusing to render a draft that can't be checked ---------------------


def test_nonexistent_path_renders_nothing(tmp_path):
    result = _call(
        to="a@b.com",
        subject="s",
        body="see attached",
        attachments=[str(tmp_path / "never_created.pdf")],
    )
    assert result["success"] is False
    assert "rendered" not in result, "no draft may be shown with a broken attachment"
    assert "does not exist" in " ".join(result["failures"])


def test_one_bad_path_fails_the_whole_draft(pdfs, tmp_path):
    """No partial success: a 6-of-7 draft is the original failure, softened."""
    result = _call(
        to="a@b.com",
        subject="s",
        body="see attached",
        attachments=pdfs + [str(tmp_path / "missing.pdf")],
    )
    assert result["success"] is False
    assert "rendered" not in result


def test_relative_path_is_refused(pdfs):
    result = _call(
        to="a@b.com", subject="s", body="see attached",
        attachments=["./4063_minutes.pdf"],
    )
    assert result["success"] is False
    assert "absolute" in " ".join(result["failures"])


def test_zero_byte_file_is_refused(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    result = _call(
        to="a@b.com", subject="s", body="see attached", attachments=[str(empty)]
    )
    assert result["success"] is False
    assert "zero bytes" in " ".join(result["failures"])


def test_body_naming_a_document_with_no_attachments_is_refused():
    """The 2026-09-08 defect itself."""
    result = _call(
        to="hk@kotoholdings.com",
        subject="Chita 4063 — handover",
        body=(
            "Attached: `4063_survey_CAD_260819.zip`, `4063_minutes_20260804.xlsx`, "
            "`4063_workproduct_contents_EN.pdf`."
        ),
        attachments=[],
    )
    assert result["success"] is False
    assert "naming a file is not showing it" in result["error"].lower()


def test_backticks_do_not_launder_a_missing_attachment():
    """Backticked names were what the old rule asked for; they prove nothing."""
    result = _call(
        to="a@b.com", subject="s",
        body="Please see `4063_RFQ_template_JP.pdf` attached.", attachments=[],
    )
    assert result["success"] is False


def test_body_with_no_document_reference_needs_no_attachment():
    """Plain correspondence must still work."""
    result = _call(
        to="a@b.com", subject="Scheduling", body="Can we meet Thursday?",
        attachments=[],
    )
    assert result["success"] is True
    assert result["attachment_count"] == 0
    assert "MEDIA:" not in result["rendered"]


def test_missing_required_fields_are_named():
    result = _call(to="", subject="", body="x")
    assert result["success"] is False
    assert "to" in result["error"] and "subject" in result["error"]


# --- draft identity -------------------------------------------------------


def test_draft_is_persisted_and_loadable_by_id(pdfs):
    result = _call(
        to="a@b.com", subject="s", body="see attached", attachments=pdfs
    )
    loaded = load_draft(result["draft_id"])
    assert loaded is not None
    assert [a["path"] for a in loaded["attachments"]] == pdfs
    assert loaded["subject"] == "s"


def test_draft_ids_are_unique(pdfs):
    a = _call(to="a@b.com", subject="s", body="b", attachments=pdfs[:1])
    b = _call(to="a@b.com", subject="s", body="b", attachments=pdfs[:1])
    assert a["draft_id"] != b["draft_id"]


def test_unknown_or_malformed_draft_id_loads_nothing():
    assert load_draft("draft_does_not_exist") is None
    assert load_draft("../../etc/passwd") is None
    assert load_draft("") is None


# --- filename detection ---------------------------------------------------


def test_filenames_are_detected_with_or_without_backticks():
    found = filenames_mentioned_in_text(
        "See `4063_minutes_20260818.pdf` and 4063_flowchart_20260901.xlsx."
    )
    assert "4063_minutes_20260818.pdf" in found
    assert "4063_flowchart_20260901.xlsx" in found


def test_prose_without_a_document_extension_is_not_a_filename():
    assert filenames_mentioned_in_text("Discussed the survey and the estimate.") == []


# --- outbound lint --------------------------------------------------------


def test_lint_blocks_a_draft_naming_files_with_none_attached():
    reply = (
        "**To:** hk@kotoholdings.com\n"
        "**Subject:** Chita 4063 — Handover\n\n"
        "Attached are `4063_survey_CAD_260819.zip` and "
        "`4063_workproduct_contents_EN.pdf`."
    )
    error = lint_outbound_reply(reply)
    assert error is not None
    assert "naming a file is not showing it" in error.lower()


def test_lint_passes_a_draft_with_media_lines():
    reply = (
        "**To:** hk@kotoholdings.com\n"
        "**Subject:** Chita 4063\n\n"
        "Attached is the contents sheet.\n\n"
        "MEDIA:/Users/davidsperling/CEJ_Files/current/solar/site_estimates/x.pdf"
    )
    assert lint_outbound_reply(reply) is None


def test_lint_ignores_non_draft_prose_mentioning_a_file():
    """Discussing a file is not presenting a draft; must not be blocked."""
    assert lint_outbound_reply("I read 4063_minutes_20260804.xlsx — page 2 is wrong.") is None


def test_lint_ignores_empty_and_whitespace_replies():
    assert lint_outbound_reply("") is None
    assert lint_outbound_reply("   \n  ") is None


def test_lint_output_names_the_correct_remedy():
    reply = "**To:** a@b.com\n**Subject:** s\n\nSee `report.pdf`."
    error = lint_outbound_reply(reply)
    assert error is not None
    assert "present_draft" in error
