"""The research guard: the tracked tree and the built artifacts carry no research material.

The standing rule is that no research project may be available in the public
repository. ``tools/research_guard.py`` enforces it three ways, and this file is
the guard's own test suite plus the one test that runs it over the real tree:

1. **Path check.** No tracked path may start with one of a short list of generic
   prefixes (the records directories, the stale playground build, ...), and none
   may be a path that moved out of the tree, which is matched by the digest of
   the exact path from a generated file, so that file lists nothing.
2. **Machine-path check.** No tracked text file may contain a path that exists
   on exactly one machine: ``/Volumes/``, ``/Users/<name>/`` or ``RESEARCH/``.
3. **Hashed-name check.** Project names are never written down in this
   repository: the guard holds only the SHA-256 digests of a private list of
   distinctive identifiers, and a match reports the file and the first twelve
   hex characters of the digest, never the token.

Almost everything below is a *positive control*. A guard that has silently
stopped matching passes every scan of every tree, which is indistinguishable
from a clean tree, so each behaviour is shown to fire on a hand-built input
before the real tree is trusted to be clean. The only plaintext token in this
repository is ``SENTINEL_TOKEN``, whose digest is in the frozen set so that the
hashed check has something it is guaranteed to find.
"""
from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import hashlib
import importlib
import io
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tools.research_guard as guard
from tools.research_guard import (
    FORBIDDEN_PREFIXES,
    MACHINE_PATTERNS,
    MAX_TOKEN_SCAN_BYTES,
    MACHINE_PATH_EXEMPT,
    MAX_TOKEN_WORDS,
    SELF_EXEMPT,
    SENTINEL_PATH,
    SENTINEL_TOKEN,
    TOKEN_DIGESTS,
    Finding,
    archive_entries,
    git_entries,
    main,
    scan_entries,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_inherited_git_environment(monkeypatch):
    """A git hook exports ``GIT_DIR`` / ``GIT_INDEX_FILE``, which would redirect
    every ``git`` call below (the scratch repositories included) to the wrong
    repository."""
    for name in [n for n in os.environ if n.startswith("GIT_")]:
        monkeypatch.delenv(name)


def _kinds(findings):
    return [f.kind for f in findings]


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------------------------------------------------------------------
# helpers that build the inputs the guard reads
# ---------------------------------------------------------------------------

def _wheel(path: Path, members: dict[str, bytes], directories: tuple[str, ...] = ()) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in directories:
            zf.writestr(name.rstrip("/") + "/", b"")
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _sdist(path: Path, members: dict[str, bytes], directories: tuple[str, ...] = ()) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name in directories:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _scratch_repo(root: Path, files: dict[str, bytes]) -> Path:
    """A throwaway repository whose index holds ``files`` (nothing committed)."""
    root.mkdir()
    _git(root, "init", "-q")
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _git(root, "add", "--", *files)
    return root


# ---------------------------------------------------------------------------
# 1. the path check
# ---------------------------------------------------------------------------

#: The generic prefixes the path check must forbid, written out here rather
#: than read back from the module so that a dropped entry fails a test instead
#: of quietly narrowing the guard.
_SPEC_PREFIXES = (
    "reviews/",
    "research_output/",
    "techreport/",
    "docs/research/",
    "docs/plans/",
    "docs/specs/",
    "docs/superpowers/",
    "playground/dist/",
)


def test_a_finding_is_an_immutable_value():
    finding = Finding("a", "b", "c")
    assert finding == Finding("a", "b", "c") and hash(finding) == hash(Finding("a", "b", "c"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        finding.path = "d"


def test_forbidden_prefix_is_found():
    found = scan_entries([("reviews/x.md", b"")])
    assert [(f.path, f.kind) for f in found] == [("reviews/x.md", "forbidden path")]


def test_the_prefix_list_is_exactly_the_generic_one():
    """A prefix that named a project would put the name in public code. The
    ``tools/run_<name>_*.py`` scripts are covered by the token in their names."""
    assert sorted(FORBIDDEN_PREFIXES) == sorted(_SPEC_PREFIXES)


@pytest.mark.parametrize("script", [
    f"tools/run_{SENTINEL_TOKEN}_phase3.py",
    f"tools/run_{SENTINEL_TOKEN}_lp.py",
    f"tools/run_{SENTINEL_TOKEN}.py",
])
def test_a_run_script_is_caught_by_the_token_in_its_name(script):
    """Word-group matching finds a token inside ``run_<token>_<rest>``, so no
    prefix is needed to forbid the scripts."""
    found = scan_entries([(script, b"")])
    assert _kinds(found) == ["hashed token"]
    assert found[0].detail == _digest(SENTINEL_TOKEN)[:12]


@pytest.mark.parametrize("prefix", _SPEC_PREFIXES)
def test_every_spec_prefix_is_forbidden(prefix):
    assert prefix in FORBIDDEN_PREFIXES
    found = scan_entries([(prefix + "x.py", b"")])
    assert found == [Finding(prefix + "x.py", "forbidden path", prefix)]


@pytest.mark.parametrize("path", [
    "docs/reviews/x.md",          # a prefix only counts at the start of a path
    "puremacro/trade_utils.py",   # the trailing slash is part of the prefix
    "docs/researchers.md",
    "tools/run_all.py",
])
def test_paths_that_only_resemble_a_prefix_pass(path):
    assert scan_entries([(path, b"")]) == []


@pytest.mark.parametrize("path", ["Reviews/x.md", "REVIEWS/x.md", "reviews\\x.md"])
def test_prefix_check_ignores_case_and_separator_style(path):
    """macOS and Windows resolve these to the same directory as ``reviews/``."""
    assert _kinds(scan_entries([(path, b"")])) == ["forbidden path"]


def test_tracked_playground_build_is_forbidden():
    """The built playground holds copies of notebooks, so it may not be tracked."""
    found = scan_entries([("playground/dist/x.js", b"")])
    assert _kinds(found) == ["forbidden path"]


def test_playground_build_gets_no_token_scan_exclusion():
    """No token-scan exclusion: one for the build would hide the notebook copies in it."""
    found = scan_entries([("playground/dist/x.js", SENTINEL_TOKEN.encode())])
    assert sorted(_kinds(found)) == ["forbidden path", "hashed token"]
    assert not hasattr(guard, "TOKEN_SCAN_EXCLUDE")


# The second clause of the path check: the paths that moved out of the tree must
# never come back. They are matched by the SHA-256 of the exact path, from a
# generated file of digests, so that this repository does not list what moved.

_PATH_VARIANTS = {
    "one character short": lambda p: p[:-1],
    "one character long": lambda p: p + "x",
    "one character different": lambda p: p[:-1] + "T",
    "one character different in the directory": lambda p: p.replace("zz/", "zy/"),
    "upper case": lambda p: p.upper(),
}


def test_a_moved_path_is_found_by_its_digest_alone():
    found = scan_entries([(SENTINEL_PATH, b"")])
    assert found == [Finding(SENTINEL_PATH, "forbidden path", "moved path")]


@pytest.mark.parametrize("variant", sorted(_PATH_VARIANTS))
def test_a_path_one_character_off_is_not_a_moved_path(variant):
    path = _PATH_VARIANTS[variant](SENTINEL_PATH)
    assert path != SENTINEL_PATH
    assert scan_entries([(path, b"")]) == []


def test_a_moved_path_is_found_whichever_way_its_separators_are_written():
    path = SENTINEL_PATH.replace("/", "\\")
    assert _kinds(scan_entries([(path, b"")])) == ["forbidden path"]


def test_a_moved_path_under_a_forbidden_prefix_is_reported_once(monkeypatch):
    """Both rules match the sentinel path: its digest is in the file, and the prefix
    list is widened to cover its directory."""
    monkeypatch.setattr(guard, "FORBIDDEN_PREFIXES", guard.FORBIDDEN_PREFIXES + ("zz/",))
    assert scan_entries([(SENTINEL_PATH, b"")]) == [Finding(SENTINEL_PATH, "forbidden path", "zz/")]


def test_the_moved_path_digest_set_is_read_when_scanning(monkeypatch):
    """So a test can substitute one, as it can the token digests."""
    monkeypatch.setattr(guard, "FORBIDDEN_PATH_DIGESTS", frozenset({_digest("zz/dummy-path.txt")}))
    assert _kinds(scan_entries([("zz/dummy-path.txt", b"")])) == ["forbidden path"]
    assert scan_entries([(SENTINEL_PATH, b"")]) == []


def test_the_moved_path_digest_file_is_sorted_unique_hex_and_not_vacuous():
    text = guard.PATH_DIGESTS_FILE.read_text(encoding="ascii")
    lines = text.splitlines()
    assert text.endswith("\n")
    assert all(re.fullmatch(r"[0-9a-f]{64}", line) for line in lines)
    assert lines == sorted(set(lines)), "the file is not sorted and unique"
    assert len(lines) > 500, f"only {len(lines)} digests: regenerated from a truncated list?"
    assert _digest(SENTINEL_PATH) in lines
    assert guard.FORBIDDEN_PATH_DIGESTS == frozenset(lines)


def test_the_moved_path_digest_file_trips_no_content_check():
    """It is hex and nothing else, so the machine and token checks find nothing in it."""
    data = guard.PATH_DIGESTS_FILE.read_bytes()
    assert scan_entries([("tools/research_guard_paths.txt", data)]) == []


def test_a_digest_file_that_lacks_the_sentinel_is_refused(tmp_path):
    """A truncated or empty file would silently switch the check off."""
    for name, text in (("empty.txt", ""), ("other.txt", "a" * 64 + "\n")):
        path = tmp_path / name
        path.write_text(text, encoding="ascii")
        with pytest.raises(ValueError, match="sentinel"):
            guard._load_path_digests(path)
    ok = tmp_path / "ok.txt"
    ok.write_text(_digest(SENTINEL_PATH) + "\n", encoding="ascii")
    assert guard._load_path_digests(ok) == frozenset({_digest(SENTINEL_PATH)})


def test_a_malformed_digest_file_is_refused():
    good = "a" * 64 + "\n" + "b" * 64 + "\n"
    assert guard._parse_digests(good) == frozenset({"a" * 64, "b" * 64})
    for bad in ("not a digest\n", "A" * 64 + "\n", "a" * 63 + "\n", "a" * 65 + "\n", good + "<<<<<<< HEAD\n"):
        with pytest.raises(ValueError):
            guard._parse_digests(bad)


# The trade package may come back under ``puremacro/trade/`` with clean files, so
# that directory is no longer a forbidden prefix; the research files that lived
# in it stay banned one exact path at a time, by digest.

#: Two research files of the trade package that must never return, whatever
#: is allowed back around them. Only generic names are spelled here; the
#: maintainer's private guard checks the full list.
_TRADE_RESEARCH_PATHS = (
    "puremacro/trade/tables.py",
    "puremacro/trade/extensions.py",
)


@pytest.mark.parametrize("path", _TRADE_RESEARCH_PATHS)
def test_trade_research_files_stay_banned(path, tmp_path):
    repo = _scratch_repo(tmp_path / "r", {path: b""})
    found = scan_entries(git_entries(repo))
    assert Finding(path, "forbidden path", "moved path") in found


def test_clean_trade_path_allowed(tmp_path):
    repo = _scratch_repo(tmp_path / "r", {"puremacro/trade/__init__.py": b""})
    assert list(git_entries(repo)) == [("puremacro/trade/__init__.py", b"")]
    assert scan_entries(git_entries(repo)) == []


# ---------------------------------------------------------------------------
# 2. the machine-path check
# ---------------------------------------------------------------------------

def test_machine_path_is_found():
    found = scan_entries([("docs/x.md", b"see /Volumes/DATA/x")])
    assert _kinds(found) == ["machine path: volume path"]


def test_the_machine_patterns_are_the_three_generic_ones():
    assert set(MACHINE_PATTERNS) == {"volume path", "user home", "research root"}


@pytest.mark.parametrize("name, text", [
    ("volume path", b"see /Volumes/DATA/x"),
    ("user home", b"open /Users/someone/notes.txt"),
    ("research root", b"cd ~/Documents/RESEARCH/proj"),
])
def test_each_machine_pattern_fires(name, text):
    assert _kinds(scan_entries([("docs/x.md", text)])) == [f"machine path: {name}"]


@pytest.mark.parametrize("text", [
    b"/Users/<name>/Documents",           # a placeholder names no machine
    b"see /Volume/x and /Volumes",        # singular, or no trailing slash
    b"SEARCH/x MYRESEARCH/x research/x",  # needs a word boundary and capitals
])
def test_text_that_is_not_a_machine_path_passes(text):
    assert scan_entries([("docs/x.md", text)]) == []


def test_machine_path_finding_names_the_line_and_not_the_path():
    text = b"ok\nok\nsee /Volumes/DATA/somewhere_private\n"
    assert scan_entries([("docs/x.md", text)]) == [
        Finding("docs/x.md", "machine path: volume path", "line 3")]


# ---------------------------------------------------------------------------
# 3. the hashed-name check
# ---------------------------------------------------------------------------

def test_sentinel_digest_is_frozen_into_the_digest_set():
    assert _digest(SENTINEL_TOKEN) in TOKEN_DIGESTS


def test_the_digest_set_is_well_formed():
    """A truncated or upper-case digest would never match anything."""
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in TOKEN_DIGESTS)
    assert len(TOKEN_DIGESTS) > 10, "regenerated from a truncated list?"


def test_hashed_token_in_content_is_found_without_revealing_it():
    found = scan_entries([("docs/x.md", f"uses {SENTINEL_TOKEN}.".encode())])
    assert _kinds(found) == ["hashed token"]
    assert SENTINEL_TOKEN not in found[0].detail
    assert found[0].detail == _digest(SENTINEL_TOKEN)[:12]
    assert SENTINEL_TOKEN not in str(found[0])


def test_hashed_token_in_a_path_is_found():
    found = scan_entries([(f"docs/{SENTINEL_TOKEN}_notes.md", b"")])
    assert _kinds(found) == ["hashed token"]
    assert SENTINEL_TOKEN not in found[0].detail


def test_a_token_in_a_mixed_case_path_is_found():
    """Tokens are lower-cased in paths as they are in content."""
    path = f"docs/{SENTINEL_TOKEN.capitalize()}-Notes.md"
    assert _kinds(scan_entries([(path, b"")])) == ["hashed token"]


@pytest.mark.parametrize("text", [
    f"def test_{SENTINEL_TOKEN}_x(): pass",    # inside a longer snake_case name
    f"{SENTINEL_TOKEN}_data = 1",
    f"x_{SENTINEL_TOKEN}",
    f"path/to/{SENTINEL_TOKEN}.py",
    f"{SENTINEL_TOKEN}-notes",                  # a hyphen ends the run
    SENTINEL_TOKEN.upper(),                     # tokens are lower-cased
    f"_{SENTINEL_TOKEN}_",
])
def test_a_token_is_found_as_a_whole_word_of_a_name(text):
    assert _kinds(scan_entries([("docs/x.md", text.encode())])) == ["hashed token"]


@pytest.mark.parametrize("text", [
    f"{SENTINEL_TOKEN}s",                  # a longer word is a different word
    f"x{SENTINEL_TOKEN}",
    f"{SENTINEL_TOKEN}1",
    SENTINEL_TOKEN[:-1],                   # a prefix of the token
])
def test_a_token_is_not_found_inside_a_longer_word(text):
    assert scan_entries([("docs/x.md", text.encode())]) == []


def test_findings_for_one_file_come_out_in_a_fixed_order():
    """Path, then the machine patterns in their declared order, then tokens."""
    path = "reviews/x.md"
    text = f"{SENTINEL_TOKEN} RESEARCH/a /Users/someone/b /Volumes/c".encode()
    assert _kinds(scan_entries([(path, text)])) == [
        "forbidden path",
        "machine path: volume path",
        "machine path: user home",
        "machine path: research root",
        "hashed token",
    ]
    assert list(MACHINE_PATTERNS) == ["volume path", "user home", "research root"]


def test_a_token_is_reported_once_per_file_however_often_it_occurs():
    entry = (f"docs/{SENTINEL_TOKEN}.md",
             f"{SENTINEL_TOKEN} {SENTINEL_TOKEN} test_{SENTINEL_TOKEN}".encode())
    assert len(scan_entries([entry])) == 1


def test_several_tokens_are_reported_in_digest_order_and_never_named(monkeypatch):
    """Six tokens, so an unsorted report matches by luck once in 720 runs."""
    words = ("alpha_beta", "gamma", "delta", "epsilon_zeta_eta", "theta", "iota")
    digests = frozenset(_digest(w) for w in words)
    monkeypatch.setattr(guard, "TOKEN_DIGESTS", digests)
    found = scan_entries([("docs/x.md", " then ".join(reversed(words)).encode())])
    assert [f.detail for f in found] == sorted(d[:12] for d in digests)
    assert not any(w in str(f) for f in found for w in words)


def test_a_token_longer_than_the_word_limit_is_found_only_as_a_whole_name(monkeypatch):
    word_group = "_".join(f"w{i}" for i in range(MAX_TOKEN_WORDS))
    longer = word_group + "_w_extra"
    monkeypatch.setattr(guard, "TOKEN_DIGESTS", frozenset({_digest(word_group), _digest(longer)}))
    embedded = scan_entries([("a.md", f"x_{word_group}_y".encode())])
    assert [f.detail for f in embedded] == [_digest(word_group)[:12]]
    whole = scan_entries([("a.md", longer.encode())])
    assert _digest(longer)[:12] in [f.detail for f in whole]
    # one word more than the limit, embedded in something longer: out of reach
    too_long = "_".join(f"w{i}" for i in range(MAX_TOKEN_WORDS + 1))
    monkeypatch.setattr(guard, "TOKEN_DIGESTS", frozenset({_digest(too_long)}))
    assert scan_entries([("a.md", f"x_{too_long}_y".encode())]) == []
    assert _kinds(scan_entries([("a.md", too_long.encode())])) == ["hashed token"]


# ---------------------------------------------------------------------------
# which files get the text checks
# ---------------------------------------------------------------------------

def test_binary_content_skips_text_checks():
    assert scan_entries([("x.bin", b"\0" + SENTINEL_TOKEN.encode())]) == []
    assert scan_entries([("y.bin", b"\0 /Volumes/A/")]) == []


def test_undecodable_bytes_in_a_path_or_in_content_do_not_stop_the_scan():
    latin_1 = b"caf\xe9 /Volumes/A/ " + SENTINEL_TOKEN.encode()
    assert _kinds(scan_entries([("docs/x.txt", latin_1)])) == [
        "machine path: volume path", "hashed token"]
    assert scan_entries([("docs/\udcff.md", b"")]) == []     # what os.fsdecode makes of a bad name


def test_only_a_nul_byte_in_the_first_8_kb_makes_a_file_binary():
    filler = b" " * 8191
    nul_inside = filler + b"\0" + b" /Volumes/A/"        # NUL at offset 8191
    nul_outside = filler + b" " + b"\0 /Volumes/A/"      # NUL at offset 8192
    assert scan_entries([("a.bin", nul_inside)]) == []
    assert _kinds(scan_entries([("b.txt", nul_outside)])) == ["machine path: volume path"]


def _padded(size: int) -> bytes:
    head = b"/Volumes/A/\n" + SENTINEL_TOKEN.encode() + b"\n"
    return head + b" " * (size - len(head))


def test_the_token_scan_stops_just_over_the_size_limit_but_the_machine_scan_does_not():
    """A file over 10 MB is covered by the path and machine-path checks only."""
    at_limit = scan_entries([("docs/a.txt", _padded(MAX_TOKEN_SCAN_BYTES))])
    over_limit = scan_entries([("docs/b.txt", _padded(MAX_TOKEN_SCAN_BYTES + 1))])
    assert _kinds(at_limit) == ["machine path: volume path", "hashed token"]
    assert _kinds(over_limit) == ["machine path: volume path"]


def test_guard_sources_are_exempt_from_machine_patterns():
    assert scan_entries([("tools/research_guard.py", b'r"/Volumes/"')]) == []


def test_guard_sources_are_exempt_from_token_scan():
    """From the sentinel's digest only: by construction they hold its plaintext.
    Every other token is still found in them (the next test)."""
    for path in ("tools/research_guard.py", "tests/test_no_research_content.py"):
        assert scan_entries([(path, SENTINEL_TOKEN.encode())]) == []


def test_a_non_sentinel_token_is_found_in_a_guard_source(monkeypatch):
    """A real name pasted into the guard or its tests must not hide there. Built
    without a real token: the digest of a second dummy token joins the set."""
    dummy = "second_dummy_token"
    monkeypatch.setattr(guard, "TOKEN_DIGESTS", guard.TOKEN_DIGESTS | {_digest(dummy)})
    text = f"{SENTINEL_TOKEN} and {dummy}".encode()
    for path in SELF_EXEMPT:
        found = scan_entries([(path, text)])
        assert [(f.kind, f.detail) for f in found] == [("hashed token", _digest(dummy)[:12])]
    # anywhere else the sentinel is reported as well
    assert [f.detail for f in scan_entries([("docs/x.md", text)])] == sorted(
        [_digest(SENTINEL_TOKEN)[:12], _digest(dummy)[:12]])


def test_the_exempt_sets_are_pinned():
    assert SELF_EXEMPT == ("tools/research_guard.py", "tests/test_no_research_content.py")
    assert MACHINE_PATH_EXEMPT == SELF_EXEMPT + ("tests/test_no_machine_specific_paths.py",)


def test_the_older_machine_path_test_is_exempt_from_machine_patterns_only():
    """It holds machine-path regex literals but no plaintext sentinel."""
    path = "tests/test_no_machine_specific_paths.py"
    text = b'_P = Path("/Users/someone/x")  # /Volumes/y RESEARCH/z'
    assert scan_entries([(path, text)]) == []
    assert _kinds(scan_entries([(path, SENTINEL_TOKEN.encode())])) == ["hashed token"]


def test_the_exemptions_never_cover_the_path(monkeypatch):
    monkeypatch.setattr(guard, "SELF_EXEMPT", ("reviews/x.md",))
    monkeypatch.setattr(guard, "MACHINE_PATH_EXEMPT", ("reviews/x.md",))
    found = scan_entries([("reviews/x.md", f"/Volumes/A/ {SENTINEL_TOKEN}".encode())])
    assert _kinds(found) == ["forbidden path"]


# ---------------------------------------------------------------------------
# built artifacts
# ---------------------------------------------------------------------------

def test_artifact_members_are_scanned(tmp_path):
    wheel = _wheel(tmp_path / "puremacro-5.0.0a1-py3-none-any.whl",
                   {"puremacro/reviews_x.py": b"/Volumes/A/"})
    found = scan_entries(archive_entries(wheel))
    assert [(f.path, f.kind) for f in found] == [
        ("puremacro/reviews_x.py", "machine path: volume path")]

    sdist = _sdist(tmp_path / "puremacro-5.0.0a1.tar.gz",
                   {"puremacro-5.0.0a1/reviews/a.md": b""})
    found = scan_entries(archive_entries(sdist))
    assert [(f.path, f.kind) for f in found] == [("reviews/a.md", "forbidden path")]


def test_wheel_members_are_read_as_they_are_and_directories_are_skipped(tmp_path):
    members = {"puremacro/__init__.py": b"A",
               "puremacro-5.0.0a1.dist-info/METADATA": b"B"}
    wheel = _wheel(tmp_path / "w.whl", members, directories=("puremacro/sub",))
    assert list(archive_entries(wheel)) == list(members.items())


def test_the_sdist_root_directory_is_stripped_and_directories_are_skipped(tmp_path):
    sdist = _sdist(
        tmp_path / "puremacro-5.0.0a1.tar.gz",
        {"puremacro-5.0.0a1/pyproject.toml": b"C",
         "puremacro-5.0.0a1/puremacro/__init__.py": b"D"},
        directories=("puremacro-5.0.0a1", "puremacro-5.0.0a1/puremacro"),
    )
    assert dict(archive_entries(sdist)) == {
        "pyproject.toml": b"C", "puremacro/__init__.py": b"D"}


def _link_sdist(path: Path, name: str, kind: bytes, linkname: str) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(f"puremacro-5.0.0a1/{name}")
        info.type = kind
        info.linkname = linkname
        tf.addfile(info)
    return path


def test_a_link_member_of_an_sdist_still_has_its_path_checked(tmp_path):
    sdist = _link_sdist(tmp_path / "puremacro-5.0.0a1.tar.gz", "reviews/link", tarfile.SYMTYPE, "elsewhere")
    assert list(archive_entries(sdist)) == [("reviews/link", b"elsewhere")]
    assert _kinds(scan_entries(archive_entries(sdist))) == ["forbidden path"]


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE], ids=["symlink", "hard link"])
def test_the_link_name_of_an_sdist_member_is_scanned_as_content(tmp_path, kind):
    """A link is published as its target text, which can name a machine."""
    sdist = _link_sdist(tmp_path / "puremacro-5.0.0a1.tar.gz", "docs/link", kind, "/Users/someone/data/")
    assert list(archive_entries(sdist)) == [("docs/link", b"/Users/someone/data/")]
    found = scan_entries(archive_entries(sdist))
    assert [(f.path, f.kind) for f in found] == [("docs/link", "machine path: user home")]


def test_a_zip_sdist_is_read_like_a_tar_sdist_and_a_wheel_is_not(tmp_path):
    members = {"puremacro-5.0.0a1/reviews/a.md": b"", "puremacro-5.0.0a1/pyproject.toml": b"x"}
    sdist = _wheel(tmp_path / "puremacro-5.0.0a1.zip", members, directories=("puremacro-5.0.0a1",))
    assert dict(archive_entries(sdist)) == {"reviews/a.md": b"", "pyproject.toml": b"x"}
    wheel = _wheel(tmp_path / "puremacro-5.0.0a1-py3-none-any.whl", members)
    assert dict(archive_entries(wheel)) == members          # a wheel's names are the installed paths
    shouting = _wheel(tmp_path / "PUREMACRO-5.0.0A1-PY3-NONE-ANY.WHL", members)
    assert dict(archive_entries(shouting)) == members


_SDIST_SUFFIXES = [".tar.gz", ".tgz", ".tar", ".zip"]


def _make_sdist(tmp_path: Path, name: str, members: dict[str, bytes]) -> Path:
    """A zip sdist if ``name`` ends in ``.zip``, else a tar sdist."""
    path = tmp_path / name
    return _wheel(path, members) if name.lower().endswith(".zip") else _sdist(path, members)


@pytest.mark.parametrize("suffix", _SDIST_SUFFIXES + [s.upper() for s in _SDIST_SUFFIXES])
def test_an_sdist_named_for_its_top_directory_is_read_whatever_its_suffix(tmp_path, suffix):
    archive = _make_sdist(tmp_path, f"puremacro-5.0.0a1{suffix}", {"puremacro-5.0.0a1/reviews/a.md": b"x"})
    assert dict(archive_entries(archive)) == {"reviews/a.md": b"x"}


@pytest.mark.parametrize("suffix", _SDIST_SUFFIXES)
def test_a_top_directory_that_is_not_the_archive_name_is_refused(tmp_path, suffix):
    """Stripped as a wrapper, a lone record directory would take its prefix with it
    and the path check would never see it; a renamed archive is as suspect. The
    message names no member."""
    cases = {
        f"puremacro-5.0.0a1{suffix}": {"reviews/a.md": b""},                 # the only directory is a record one
        f"renamed{suffix}": {"puremacro-5.0.0a1/reviews/a.md": b""},         # a proper root, but not this archive's
    }
    for name, members in cases.items():
        with pytest.raises(ValueError, match="does not match the file name") as exc:
            list(archive_entries(_make_sdist(tmp_path, name, members)))
        assert "reviews" not in str(exc.value)


@pytest.mark.parametrize("kind", ["tar", "zip"])
@pytest.mark.parametrize("members, message", [
    ({"first/x.py": b"", "second/y.py": b""}, "share one top-level directory"),
    ({"puremacro-5.0.0a1/x.py": b"", "second/y.py": b""}, "share one top-level directory"),    # one root is the right one
    ({"README": b"", "puremacro-5.0.0a1/x.py": b""}, "one of them a file"),
    ({"puremacro-5.0.0a1": b"a file, not a directory"}, "one of them a file"),                  # named like the archive
    ({"README": b""}, "one of them a file"),
], ids=["two roots", "right root beside another", "file beside a root", "file named like the archive", "only a file"])
def test_an_sdist_without_one_top_directory_is_refused(tmp_path, kind, members, message):
    """Stripping the first component of every member is only right when they all share
    it. The message gives counts, never member names."""
    name = "puremacro-5.0.0a1.zip" if kind == "zip" else "puremacro-5.0.0a1.tar.gz"
    with pytest.raises(ValueError, match=message) as exc:
        list(archive_entries(_make_sdist(tmp_path, name, members)))
    assert not any(word in str(exc.value) for word in ("first", "second", "README", "x.py", "y.py"))


def test_something_that_is_neither_wheel_nor_sdist_is_rejected(tmp_path):
    junk = tmp_path / "x.txt"
    junk.write_bytes(b"just text")
    with pytest.raises(ValueError):
        list(archive_entries(junk))


# ---------------------------------------------------------------------------
# the tracked tree
# ---------------------------------------------------------------------------

def test_git_entries_yields_tracked_files_with_their_working_tree_bytes(tmp_path):
    repo = _scratch_repo(tmp_path / "r", {"a.txt": b"staged", "sub/b.txt": b"x"})
    (repo / "a.txt").write_bytes(b"edited after staging")
    (repo / "untracked.txt").write_bytes(b"never added")
    assert list(git_entries(repo)) == [
        ("a.txt", b"edited after staging"), ("sub/b.txt", b"x")]


def test_git_entries_scans_the_whole_repository_from_a_subdirectory(tmp_path):
    """``git ls-files`` run in a subdirectory lists only that directory, relative to
    it, which would make every prefix check miss."""
    repo = _scratch_repo(tmp_path / "r", {"a.txt": b"A", "sub/b.txt": b"B", "reviews/c.md": b""})
    whole = [("a.txt", b"A"), ("reviews/c.md", b""), ("sub/b.txt", b"B")]
    assert list(git_entries(repo)) == whole
    assert list(git_entries(repo / "sub")) == whole
    assert _kinds(scan_entries(git_entries(repo / "sub"))) == ["forbidden path"]


def _symlink(link: Path, target: str) -> None:
    """Create a symlink, or skip: Windows round-trips link text differently, and git
    there checks symlinks out as plain files, which the regular-file path covers."""
    if sys.platform == "win32":
        pytest.skip("symlink text is not portable to Windows")
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create symlinks")


def _repo_with_links(root: Path, links: dict[str, str]) -> Path:
    repo = _scratch_repo(root, {"keep.txt": b"k"})
    _git(repo, "config", "core.symlinks", "true")
    for name, target in links.items():
        _symlink(repo / name, target)
    _git(repo, "add", "--", *links)
    return repo


def test_a_tracked_symlink_yields_its_link_text_not_its_target(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"TARGET BYTES " + SENTINEL_TOKEN.encode())
    repo = _repo_with_links(tmp_path / "r", {"link": str(outside)})
    entries = dict(git_entries(repo))
    assert entries["link"] == os.fsencode(os.readlink(repo / "link"))
    assert b"TARGET BYTES" not in entries["link"]
    assert scan_entries(git_entries(repo)) == []          # the target's sentinel is not this tree's


def test_a_dangling_tracked_symlink_still_has_its_path_and_text_checked(tmp_path):
    repo = _repo_with_links(tmp_path / "r", {"reviews/link": "nowhere", "docs/link": "/Volumes/DATA/x"})
    assert dict(git_entries(repo))["reviews/link"] == b"nowhere"
    found = scan_entries(git_entries(repo))
    assert [(f.path, f.kind) for f in found] == [
        ("docs/link", "machine path: volume path"), ("reviews/link", "forbidden path")]


def test_a_symlink_to_a_directory_yields_its_link_text(tmp_path):
    repo = _scratch_repo(tmp_path / "r", {"sub/b.txt": b"B"})
    _git(repo, "config", "core.symlinks", "true")
    _symlink(repo / "alias", "sub")
    _git(repo, "add", "--", "alias")
    assert dict(git_entries(repo))["alias"] == b"sub"


def test_git_entries_skips_a_tracked_path_missing_from_the_working_tree(tmp_path):
    repo = _scratch_repo(tmp_path / "r", {"keep.txt": b"k", "gone.txt": b"g"})
    (repo / "gone.txt").unlink()
    assert list(git_entries(repo)) == [("keep.txt", b"k")]


def test_git_entries_raises_outside_a_repository(tmp_path, monkeypatch):
    """A failed ``git`` call must not look like an empty, clean tree."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))   # never find a parent repo
    with pytest.raises(OSError, match="git rev-parse failed"):
        list(git_entries(tmp_path))


def _format(findings, limit=200):
    counts = collections.Counter(f.kind for f in findings)
    head = ", ".join(f"{n} x {kind}" for kind, n in sorted(counts.items()))
    lines = [str(f) for f in findings[:limit]]
    if len(findings) > limit:
        lines.append(f"... and {len(findings) - limit} more")
    return f"{len(findings)} research-guard finding(s): {head}\n" + "\n".join(lines)


def test_the_tree_has_no_research_content():
    findings = scan_entries(git_entries(REPO))
    if findings:
        pytest.fail(_format(findings), pytrace=False)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def test_main_prints_one_line_per_finding_and_returns_1(tmp_path, capsys):
    sdist = _sdist(tmp_path / "puremacro-5.0.0a1.tar.gz", {
        "puremacro-5.0.0a1/reviews/a.md": b"",
        "puremacro-5.0.0a1/docs/x.md": f"see /Volumes/A/ and {SENTINEL_TOKEN}".encode(),
    })
    assert main([str(sdist)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "reviews/a.md: forbidden path: reviews/",
        "docs/x.md: machine path: volume path: line 1",
        f"docs/x.md: hashed token: {_digest(SENTINEL_TOKEN)[:12]}",
    ]
    assert SENTINEL_TOKEN not in out


def test_main_returns_0_and_prints_no_finding_for_a_clean_artifact(tmp_path, capsys):
    wheel = _wheel(tmp_path / "puremacro-5.0.0a1-py3-none-any.whl",
                   {"puremacro/__init__.py": b"__version__ = '5.0.0a1'\n"})
    assert main([str(wheel)]) == 0
    assert capsys.readouterr().out == ""


def test_main_git_scans_the_repository_of_the_working_directory(tmp_path, monkeypatch, capsys):
    repo = _scratch_repo(tmp_path / "r", {"ok/a.py": b"x = 1\n", "reviews/b.md": b""})
    monkeypatch.chdir(repo / "ok")          # a subdirectory: the whole repo is scanned
    assert main(["--git"]) == 1
    assert capsys.readouterr().out.splitlines() == ["reviews/b.md: forbidden path: reviews/"]


def test_main_scans_the_tree_and_artifacts_in_one_run(tmp_path, monkeypatch, capsys):
    repo = _scratch_repo(tmp_path / "r", {"ok.py": b"x = 1\n"})
    wheel = _wheel(tmp_path / "w.whl", {"techreport/a.md": b""})
    monkeypatch.chdir(repo)
    assert main(["--git", str(wheel)]) == 1
    assert capsys.readouterr().out.splitlines() == ["techreport/a.md: forbidden path: techreport/"]


def test_main_with_nothing_to_scan_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_main_git_outside_a_repository_is_an_error_not_a_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))   # never find a parent repo
    monkeypatch.chdir(tmp_path)
    assert main(["--git"]) == 2
    assert capsys.readouterr().out == ""


def test_main_refuses_to_pass_a_scan_that_saw_no_files(tmp_path, capsys):
    """An empty file list passes every check, so it must not count as clean."""
    empty = _wheel(tmp_path / "empty.whl", {})
    assert main([str(empty)]) == 2
    assert capsys.readouterr().out == ""


def test_main_reports_an_artifact_it_cannot_read(tmp_path, capsys):
    assert main([str(tmp_path / "missing.whl")]) == 2
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# every internal import still resolves
# ---------------------------------------------------------------------------
#
# Removing a module leaves its importers behind. The guard's job is to keep the
# tree free of research material, and the cheapest way a removal goes wrong is a
# surviving ``from puremacro.<removed> import ...`` that nothing imports until a
# user does. tests/ is deliberately not scanned: a test that imports a removed
# module already fails loudly at collection, and the IO parity tests import a
# vendored puremacro from an external workspace on purpose.

_IMPORT_SCAN_ROOTS = ("puremacro/", "tools/", "playground/")
_IMPORT_SCAN_SKIP = ("playground/dist/",)


def _tracked_paths(repo: Path) -> frozenset[str]:
    raw = subprocess.run(["git", "ls-files", "-z", "-c", "-o", "--exclude-standard"], cwd=repo, check=True,
                         capture_output=True).stdout
    return frozenset(os.fsdecode(p) for p in raw.split(b"\0") if p)


def _is_module(dotted: str, tracked: frozenset[str]) -> bool:
    base = dotted.replace(".", "/")
    return f"{base}/__init__.py" in tracked or f"{base}.py" in tracked


def _package_of(path: str) -> list[str]:
    """The package a file under puremacro/ lives in; a package's __init__ is it."""
    return path[: -len(".py")].split("/")[:-1]


def _relative_base(package: list[str], level: int, module: str | None) -> str | None:
    """The dotted module a relative import names, or ``None`` when it climbs above
    the top-level package."""
    up = level - 1
    if up >= len(package):
        return None
    return ".".join(package[: len(package) - up] + ([module] if module else []))


def _literal_strings(value: ast.expr) -> Iterator[str]:
    """The strings in a literal list, tuple or set, or in a ``+`` of them."""
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                yield element.value
    elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        yield from _literal_strings(value.left)
        yield from _literal_strings(value.right)


def _assigned_names(target: ast.expr) -> Iterator[str]:
    """The names an assignment target binds, unpacking included. An attribute or
    an item binds none."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _assigned_names(element)
    elif isinstance(target, ast.Starred):
        yield from _assigned_names(target.value)


def _module_level(body: list[ast.stmt]) -> Iterator[ast.AST]:
    """The statements that run when a module is imported, in source order: the body
    itself and the bodies of ``if``, ``try``, ``with``, ``for``, ``while`` and
    ``match`` inside it, but not the inside of a function or a class."""
    for node in body:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)):
                yield from _module_level([child])


@dataclasses.dataclass(frozen=True)
class _InitNames:
    """What a package's ``__init__.py`` offers to ``from package import name``."""

    bound: frozenset[str]       # bound at module level; an import of the package from itself is not
    declared: tuple[str, ...]   # the entries of a literal ``__all__``, in order
    strings: frozenset[str]     # every string literal in the file; empty unless it is lazy
    lazy: bool                  # defines a module-level ``__getattr__``
    open_ended: bool            # star-imports something, so it may serve any name

    def provides(self, name: str) -> bool:
        """Whether the file offers ``name``. An ``__all__`` entry is a claim, not
        evidence: a stale one would vouch for a removed re-export. A lazy package
        serves what its file lists, so any string in it counts there, and the
        runtime test at the end of this file is the backstop for that."""
        return self.open_ended or name in self.bound or name in self.strings


def _analyse_init(tree: ast.Module, package: list[str]) -> _InitNames:
    """What the ``__init__.py`` of ``package`` (parsed into ``tree``) offers.

    A binding is an assignment target, a ``def`` or ``class``, or an ``import``,
    all at module level. An import of the package from inside its own ``__init__``
    is not evidence: ``from . import x`` binds ``x`` only if ``x`` exists, and a
    leftover re-export of a removed submodule would vouch for itself.
    """
    this = ".".join(package)
    bound: set[str] = set()
    declared: list[str] = []
    lazy = open_ended = False
    for node in _module_level(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            lazy = lazy or node.name == "__getattr__"
        elif isinstance(node, ast.Import):
            bound.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            origin = node.module if node.level == 0 else _relative_base(package, node.level, node.module)
            if origin != this:
                for alias in node.names:
                    if alias.name == "*":
                        open_ended = True
                    else:
                        bound.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "__all__" and node.value is not None:
                    declared.extend(_literal_strings(node.value))
                if node.value is not None and not isinstance(node, ast.AugAssign):
                    bound.update(_assigned_names(target))
    strings = frozenset(c.value for c in ast.walk(tree)
                        if isinstance(c, ast.Constant) and isinstance(c.value, str)) if lazy else frozenset()
    return _InitNames(frozenset(bound), tuple(declared), strings, lazy, open_ended)


@functools.lru_cache(maxsize=None)
def _init_names(root: Path, init: str) -> _InitNames | None:
    """``_analyse_init`` for the ``__init__`` at ``init`` under ``root``. ``None``
    when the file cannot be read or parsed: that says nothing about a name (a
    missing file is skipped, a broken one is reported where it is scanned)."""
    try:
        tree = ast.parse((root / init).read_bytes())
    except (OSError, SyntaxError):
        return None
    return _analyse_init(tree, init[: -len("/__init__.py")].split("/"))


def _stale_all_entries(init: str, tree: ast.Module, tracked: frozenset[str]) -> list[str]:
    """The entries of the literal ``__all__`` in the package ``__init__`` at ``init``
    that nothing provides: not bound at module level, and not a tracked submodule.
    A lazy package serves names at runtime (judged by the runtime test) and a star
    import may serve any, so neither is judged here."""
    package = init[: -len("/__init__.py")].split("/")
    names = _analyse_init(tree, package)
    if names.lazy or names.open_ended:
        return []
    dotted = ".".join(package)
    return [entry for entry in names.declared
            if entry not in names.bound and not _is_module(f"{dotted}.{entry}", tracked)]


def _unresolved_names(base: str, names: list[str], tracked: frozenset[str],
                      root: Path) -> list[str] | None:
    """The ``names`` that ``from <base> import <name>`` cannot find, or ``None`` when
    ``base`` is no module at all.

    A plain module may define any name, and ``*`` always resolves. In a package a
    name must be a submodule or something its ``__init__`` provides.
    """
    folder = base.replace(".", "/")
    init = f"{folder}/__init__.py"
    if f"{folder}.py" in tracked and init not in tracked:
        return []
    submodules = {name for name in names if _is_module(f"{base}.{name}", tracked)}
    if init not in tracked:                         # a namespace package, or nothing at all
        return [n for n in names if n not in submodules] if submodules else None
    missing = []
    for name in names:
        if name == "*" or name in submodules:
            continue
        provided = _init_names(root, init)
        if provided is not None and not provided.provides(name):
            missing.append(name)
    return missing


def _check_imports(path: str, source: bytes, tracked: frozenset[str],
                   root: Path) -> tuple[list[str], int]:
    """The ``puremacro`` imports in ``source`` that name no tracked module and no
    name their package provides, and how many it checked.

    ``from M import Y`` resolves when ``M`` is a plain module (it may define ``Y``),
    or, for a package ``M``, when ``M.Y`` is a tracked module or ``M/__init__.py``
    provides ``Y`` (see ``_provided_names``). Package ``__init__`` files are read
    from ``root``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: does not parse ({exc.msg})"], 0
    package = _package_of(path) if path.startswith("puremacro/") else None
    problems: list[str] = []
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "puremacro":
                    continue
                checked += 1
                if not _is_module(alias.name, tracked):
                    problems.append(f"{path}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            shown = "." * node.level + (node.module or "")
            if node.level == 0:
                if (node.module or "").split(".")[0] != "puremacro":
                    continue
                base = node.module
            elif package is None:
                continue               # relative imports are resolved inside puremacro/ only
            else:
                base = _relative_base(package, node.level, node.module)
                if base is None:
                    checked += 1
                    problems.append(f"{path}:{node.lineno}: from {shown} import ... "
                                    "(above the top-level package)")
                    continue
            checked += 1
            unresolved = _unresolved_names(base, [a.name for a in node.names], tracked, root)
            if unresolved is None:
                problems.append(f"{path}:{node.lineno}: from {shown} import ...")
            else:
                problems.extend(f"{path}:{node.lineno}: from {shown} import {name}"
                                for name in unresolved)
    return sorted(problems), checked


def _write_tree(root: Path, files: dict[str, str]) -> frozenset[str]:
    """Write ``files`` under ``root`` and return their paths: a tiny tree to resolve against."""
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return frozenset(files)


_FAKE_FILES = {
    "puremacro/__init__.py": "",
    "puremacro/a/__init__.py": "a_name = 1\n",
    "puremacro/a/b.py": "",
    "puremacro/a/c.py": "",
    "puremacro/top.py": "",
}


@pytest.mark.parametrize("path, source, expected", [
    # absolute imports
    ("tools/t.py", "import puremacro", []),
    ("tools/t.py", "import puremacro.a.b", []),
    ("tools/t.py", "import puremacro.a.gone", ["tools/t.py:1: import puremacro.a.gone"]),
    ("tools/t.py", "from puremacro.a import b", []),                  # a submodule
    ("tools/t.py", "from puremacro.a import a_name", []),             # a name the package binds
    ("tools/t.py", "from puremacro.a import gone", ["tools/t.py:1: from puremacro.a import gone"]),
    ("tools/t.py", "from puremacro.a import b, gone, a_name",         # each name is judged alone
     ["tools/t.py:1: from puremacro.a import gone"]),
    ("tools/t.py", "from puremacro.a import *", []),
    ("tools/t.py", "from puremacro.a.b import anything", []),         # a plain module may define any name
    ("tools/t.py", "from puremacro.gone import x", ["tools/t.py:1: from puremacro.gone import ..."]),
    ("tools/t.py", "import numpy\nfrom os import path", []),          # only puremacro is checked
    ("tools/t.py", "def f():\n    try:\n        import puremacro.a.gone\n    except ImportError:\n        pass",
     ["tools/t.py:3: import puremacro.a.gone"]),                      # nested imports count
    # relative imports inside puremacro/
    ("puremacro/a/c.py", "from . import b", []),
    ("puremacro/a/c.py", "from . import a_name", []),
    ("puremacro/a/c.py", "from . import gone", ["puremacro/a/c.py:1: from . import gone"]),
    ("puremacro/a/c.py", "from .b import f", []),
    ("puremacro/a/c.py", "from .gone import f", ["puremacro/a/c.py:1: from .gone import ..."]),
    ("puremacro/a/c.py", "from .. import top", []),
    ("puremacro/a/c.py", "from .. import gone", ["puremacro/a/c.py:1: from .. import gone"]),
    ("puremacro/a/c.py", "from ..top import f", []),
    ("puremacro/a/c.py", "from ..gone import f", ["puremacro/a/c.py:1: from ..gone import ..."]),
    ("puremacro/a/c.py", "from ... import x",
     ["puremacro/a/c.py:1: from ... import ... (above the top-level package)"]),
    ("puremacro/a/__init__.py", "from .b import f", []),              # an __init__ is its own package
    ("puremacro/top.py", "from . import a", []),
    ("puremacro/top.py", "from .. import a",
     ["puremacro/top.py:1: from .. import ... (above the top-level package)"]),
    ("tools/t.py", "from . import x", []),                            # not resolved outside puremacro/
    # a file that does not parse is a finding, not a skip
    ("tools/t.py", "def (:", ["tools/t.py:1: does not parse (invalid syntax)"]),
])
def test_the_import_resolver_flags_exactly_the_unresolvable_imports(tmp_path, path, source, expected):
    tracked = _write_tree(tmp_path, _FAKE_FILES)
    problems, _ = _check_imports(path, source.encode(), tracked, tmp_path)
    assert problems == expected


def test_the_import_resolver_counts_what_it_checks(tmp_path):
    tracked = _write_tree(tmp_path, _FAKE_FILES)
    source = b"import puremacro.a.b\nfrom . import x\nimport numpy\n"
    assert _check_imports("puremacro/a/c.py", source, tracked, tmp_path)[1] == 2


# ``from package import name`` needs the name to exist. Accepting it whenever the
# package exists would let a removed submodule through for as long as the package
# survives, which is the usual shape of a leftover import. Each case below writes
# a tiny ``puremacro/pkg/__init__.py`` and asks whether
# ``from puremacro.pkg import wanted`` resolves.

_WANTED_PROBLEM = ["tools/t.py:1: from puremacro.pkg import wanted"]


def _problems_for(root: Path, init: str, statement: str = "from puremacro.pkg import wanted") -> list[str]:
    tracked = _write_tree(root, {"puremacro/__init__.py": "", "puremacro/pkg/__init__.py": init + "\n"})
    return _check_imports("tools/t.py", statement.encode(), tracked, root)[0]


def test_a_removed_submodule_fails_and_a_name_the_package_binds_passes(tmp_path):
    tracked = _write_tree(tmp_path, {
        "puremacro/__init__.py": "",
        "puremacro/pkg/__init__.py": "def helper():\n    return 1\n",
        "puremacro/pkg/alive.py": "",
    })

    def check(statement):
        return _check_imports("tools/t.py", statement.encode(), tracked, tmp_path)[0]

    assert check("from puremacro.pkg import alive") == []          # a submodule that exists
    assert check("from puremacro.pkg import helper") == []         # a name pkg/__init__.py binds
    assert check("from puremacro.pkg import removed") == [         # a submodule that was removed
        "tools/t.py:1: from puremacro.pkg import removed"]


@pytest.mark.parametrize("init", [
    "wanted = 1",
    "first, (second, *wanted) = 1, (2, 3)",
    "wanted: int = 1",
    "def wanted(): ...",
    "async def wanted(): ...",
    "class wanted: ...",
    "import os as wanted",
    "import wanted.sub",
    "from os import wanted",
    "from os import path as wanted",
    "from .impl import wanted",                         # a re-export from a submodule
    "try:\n    from os import path as wanted\nexcept ImportError:\n    wanted = None",
    "try:\n    import os\nexcept ImportError:\n    wanted = None",     # bound only in a handler
    "if True:\n    def wanted(): ...",
])
def test_every_kind_of_module_level_binding_counts(tmp_path, init):
    assert _problems_for(tmp_path, init) == []


@pytest.mark.parametrize("init", [
    "def f():\n    wanted = 1",                  # a function body is not module level
    "class C:\n    wanted = 1",                  # nor is a class body
    "def f():\n    import wanted",
    "# wanted = 1",
    "'wanted'",                                  # a string alone serves nothing
    "obj.wanted = 1",                            # an attribute or item target binds no name
    "table['wanted'] = 1",
    "wanted: int",                               # an annotation alone binds nothing
    "wanted += 1",                               # an augmented assignment needs an earlier binding
    "other = ['wanted']",                        # a list that is not __all__
    "__all__ = sorted(['wanted'])",              # an __all__ that is not a literal
    "__all__ = ['wanted']",                      # an __all__ entry is a claim, not evidence: a stale
    "__all__ = ('wanted',)",                     # one would vouch for a removed re-export
    "__all__ = ['other'] + ['wanted']",
    "__all__ = []\n__all__ += ['wanted']",
    "__all__: list = ['wanted']",
    "_TABLE = {'wanted': 'impl'}",               # a lazy table without a module __getattr__
    "def other(): ...\n_TABLE = {'wanted': 'impl'}",                        # only __getattr__ makes it lazy
    "class C:\n    def __getattr__(self, name): ...\n_TABLE = {'wanted': 'impl'}",   # not a module __getattr__
])
def test_what_is_not_a_module_level_binding_does_not_count(tmp_path, init):
    assert _problems_for(tmp_path, init) == _WANTED_PROBLEM


def test_a_lazy_package_serves_the_names_its_file_lists(tmp_path):
    """A package with a module ``__getattr__`` serves names from a table in its own
    file (the narrative sources do: ``__all__`` is computed, not a literal)."""
    init = ("_LAZY = {'served': 'impl'}\n__all__ = sorted(_LAZY)\n\n\n"
            "def __getattr__(name):\n    raise AttributeError(name)\n")
    assert _problems_for(tmp_path, init, "from puremacro.pkg import served") == []
    assert _problems_for(tmp_path / "again", init, "from puremacro.pkg import wanted") == _WANTED_PROBLEM


def test_a_package_that_star_imports_can_provide_any_name(tmp_path):
    assert _problems_for(tmp_path, "from .impl import *") == []


def test_a_package_init_that_cannot_be_read_gives_no_verdict(tmp_path):
    unparseable = _write_tree(tmp_path, {"puremacro/__init__.py": "", "puremacro/pkg/__init__.py": "def (:\n"})
    statement = b"from puremacro.pkg import wanted\n"
    assert _check_imports("tools/t.py", statement, unparseable, tmp_path)[0] == []
    # ... but the broken file is reported where it is scanned
    assert _check_imports("puremacro/pkg/__init__.py", b"def (:\n", unparseable, tmp_path)[0]
    # tracked, yet absent from the working tree (deleted mid-edit)
    absent = tmp_path / "absent"
    absent.mkdir()
    tracked = frozenset({"puremacro/__init__.py", "puremacro/pkg/__init__.py"})
    assert _check_imports("tools/t.py", statement, tracked, absent)[0] == []


@pytest.mark.parametrize("statement", [
    "from . import wanted",
    "from . import wanted as wanted",
    "from puremacro.pkg import wanted",
    "from . import wanted\nfrom . import wanted",       # two of them do not vouch for each other
])
def test_a_package_cannot_vouch_for_a_name_it_imports_from_itself(tmp_path, statement):
    """``from . import x`` in ``pkg/__init__.py`` binds ``x`` only if ``x`` exists, so it
    is not evidence that it does. This is the shape of an ``__init__`` that still
    re-exports a removed submodule."""
    tracked = _write_tree(tmp_path, {"puremacro/__init__.py": "", "puremacro/pkg/__init__.py": statement + "\n"})
    own, _ = _check_imports("puremacro/pkg/__init__.py", (statement + "\n").encode(), tracked, tmp_path)
    other, _ = _check_imports("tools/t.py", b"from puremacro.pkg import wanted\n", tracked, tmp_path)
    assert len(own) == statement.count("import")
    assert other == _WANTED_PROBLEM


def test_a_directory_without_an_init_offers_only_its_submodules(tmp_path):
    tracked = _write_tree(tmp_path, {"puremacro/__init__.py": "", "puremacro/ns/one.py": ""})

    def check(statement):
        return _check_imports("tools/t.py", statement.encode(), tracked, tmp_path)[0]

    assert check("from puremacro.ns import one") == []
    assert check("from puremacro.ns import one, two") == ["tools/t.py:1: from puremacro.ns import two"]
    assert check("from puremacro.ns import two") == ["tools/t.py:1: from puremacro.ns import ..."]


def test_a_literal_all_entry_of_a_lazy_package_resolves(tmp_path):
    """A lazy package serves what its file lists; the runtime test below is the backstop."""
    init = "__all__ = ['served']\n\n\ndef __getattr__(name):\n    raise AttributeError(name)\n"
    assert _problems_for(tmp_path, init, "from puremacro.pkg import served") == []


def test_a_stale_all_entry_does_not_vouch_for_a_removed_re_export(tmp_path):
    """The shape of a package that re-exported from a module that was then removed:
    the import is gone, the ``__all__`` string is not."""
    init = "from puremacro.models.kept import Kept\n\n__all__ = ['Kept', 'Gone']\n"
    tracked = _write_tree(tmp_path, {"puremacro/__init__.py": "", "puremacro/models/__init__.py": init,
                                     "puremacro/models/kept.py": ""})

    def check(name):
        source = f"from puremacro.models import {name}".encode()
        return _check_imports("tools/t.py", source, tracked, tmp_path)[0]

    assert check("Kept") == []
    assert check("Gone") == ["tools/t.py:1: from puremacro.models import Gone"]
    assert _stale_all_entries("puremacro/models/__init__.py", ast.parse(init), tracked) == ["Gone"]


@pytest.mark.parametrize("init, expected", [
    ("x = 1\n__all__ = ['x']", []),
    ("__all__ = ['gone']", ["gone"]),
    ("from .impl import thing\n__all__ = ['thing']", []),
    ("def f(): ...\nclass C: ...\n__all__ = ['f', 'C', 'gone']", ["gone"]),
    ("from . import sub\n__all__ = ['sub']", []),               # sub is a tracked submodule
    ("from . import ghost\n__all__ = ['ghost']", ["ghost"]),    # a self-import binds nothing
    ("__all__ = []\n__all__ += ['gone']", ["gone"]),
    ("__all__ = ['a'] + ['gone']\na = 1", ["gone"]),
    ("__all__ = ['gone_1', 'gone_2']", ["gone_1", "gone_2"]),    # in the order listed
    ("__all__ = sorted(['gone'])", []),                          # not a literal: nothing to judge
    ("from .impl import *\n__all__ = ['anything']", []),        # a star import may serve any name
    ("__all__ = ['served']\n\ndef __getattr__(name): ...", []),  # lazy: judged at runtime instead
    ("", []),
])
def test_a_literal_all_entry_must_be_bound_or_a_submodule(init, expected):
    tracked = frozenset({"puremacro/pkg/__init__.py", "puremacro/pkg/sub.py", "puremacro/pkg/impl.py"})
    assert _stale_all_entries("puremacro/pkg/__init__.py", ast.parse(init), tracked) == expected


def test_package_all_entries_are_bound():
    """Every literal ``__all__`` entry of a puremacro package is a real attribute:
    bound at module level (a package's import of itself does not count) or a tracked
    submodule. A stale entry is what would otherwise let ``from package import
    removed`` look fine."""
    tracked = _tracked_paths(REPO)
    inits = sorted(p for p in tracked if p.startswith("puremacro/") and p.endswith("/__init__.py"))
    stale: list[str] = []
    entries = 0
    for init in inits:
        try:
            tree = ast.parse((REPO / init).read_bytes())
        except (OSError, SyntaxError):
            continue                    # deleted mid-edit, or reported where it is scanned
        package = init[: -len("/__init__.py")].split("/")
        names = _analyse_init(tree, package)
        if not (names.lazy or names.open_ended):
            entries += len(names.declared)
        stale += [f"{init}: __all__ lists {entry!r}, which nothing binds"
                  for entry in _stale_all_entries(init, tree, tracked)]
    assert len(inits) > 30 and entries > 500, (
        f"the scan is vacuous: {len(inits)} packages, {entries} __all__ entries judged")
    assert not stale, f"{len(stale)} stale __all__ entr(ies):\n  " + "\n  ".join(stale)


# ---------------------------------------------------------------------------
# lazy packages: the runtime backstop
# ---------------------------------------------------------------------------
#
# The resolver above trusts every string a lazy package (one that defines a
# module ``__getattr__``) lists. That is only as good as the list, so each such
# package is imported and every name in its runtime ``__all__`` is read for real.
# A module missing for an optional third-party dependency is tolerated and
# reported in the skip reason; a missing ``puremacro`` module, or an
# ``AttributeError``, is a stale entry and fails.

def _is_optional_dependency(exc: ImportError) -> bool:
    """Whether the import failed on a module outside puremacro: an optional
    third-party dependency that is not installed."""
    missing = exc.name or ""
    return bool(missing) and missing != "puremacro" and not missing.startswith("puremacro.")


def _lazy_all_problems(module) -> tuple[list[str], list[str]]:
    """``(failures, tolerated)`` from reading every name in the module's runtime
    ``__all__``. It is the runtime value that counts: some packages compute it, so
    it is not in the source to be read."""
    failures: list[str] = []
    tolerated: list[str] = []
    for name in getattr(module, "__all__", ()):
        try:
            getattr(module, name)
        except ImportError as exc:
            if _is_optional_dependency(exc):
                tolerated.append(f"{name} (needs {exc.name})")
            else:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        except Exception as exc:        # an AttributeError, or anything else a lazy import raises
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures, tolerated


def _lazy_packages(repo: Path) -> list[str]:
    """The dotted names of the packages under ``repo/puremacro`` whose ``__init__``
    defines a module-level ``__getattr__``."""
    found = []
    for init in sorted((repo / "puremacro").rglob("__init__.py")):
        try:
            tree = ast.parse(init.read_bytes())
        except (OSError, SyntaxError):
            continue
        package = init.relative_to(repo).parent.parts
        if _analyse_init(tree, list(package)).lazy:
            found.append(".".join(package))
    return found


_LAZY_PACKAGES = _lazy_packages(REPO)


@pytest.mark.parametrize("exc, tolerated", [
    (ModuleNotFoundError("No module named 'bs4'", name="bs4"), True),
    (ImportError("cannot import name 'x' from 'bs4'", name="bs4"), True),
    (ModuleNotFoundError("No module named 'puremacrofoo'", name="puremacrofoo"), True),   # another top-level module
    (ModuleNotFoundError("No module named 'puremacro.gone'", name="puremacro.gone"), False),
    (ImportError("cannot import name 'x' from 'puremacro.a'", name="puremacro.a"), False),
    (ImportError("cannot import name 'x' from 'puremacro'", name="puremacro"), False),
    (ImportError("no module recorded"), False),
])
def test_only_a_missing_third_party_module_is_tolerated(exc, tolerated):
    assert _is_optional_dependency(exc) is tolerated


_LAZY_BODY = """\
__all__ = ["served", "stale", "needs_optional", "needs_ours", "broken"]


def __getattr__(name):
    if name == "served":
        return 1
    if name == "needs_optional":
        raise ModuleNotFoundError("No module named 'some_optional_dependency'", name="some_optional_dependency")
    if name == "needs_ours":
        raise ModuleNotFoundError("No module named 'puremacro.gone'", name="puremacro.gone")
    if name == "broken":
        raise RuntimeError("import-time failure")
    raise AttributeError(name)
"""


@pytest.fixture
def make_package(tmp_path, monkeypatch):
    """Write and import a throwaway package, and forget it afterwards."""
    made: list[str] = []

    def make(name: str, init: str):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "__init__.py").write_text(init, encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        made.append(name)
        return importlib.import_module(name)

    yield make
    for name in made:
        sys.modules.pop(name, None)


def test_the_runtime_backstop_fails_stale_entries_and_tolerates_optional_dependencies(make_package):
    failures, tolerated = _lazy_all_problems(make_package("lazy_fixture_a", _LAZY_BODY))
    assert sorted(f.split(":")[0] for f in failures) == ["broken", "needs_ours", "stale"]
    assert tolerated == ["needs_optional (needs some_optional_dependency)"]


def test_the_runtime_backstop_passes_a_package_that_serves_every_name(make_package):
    module = make_package("lazy_fixture_b", '__all__ = ["a", "b"]\n\n\ndef __getattr__(name):\n    return name\n')
    assert _lazy_all_problems(module) == ([], [])


def test_the_runtime_backstop_reads_the_runtime_all_not_the_literal(make_package):
    """The narrative sources compute ``__all__``; only the runtime value shows it."""
    module = make_package(
        "lazy_fixture_c",
        "_NAMES = {'x': 1}\n__all__ = sorted(_NAMES)\n\n\ndef __getattr__(name):\n    raise AttributeError(name)\n")
    failures, _ = _lazy_all_problems(module)
    assert [f.split(":")[0] for f in failures] == ["x"]


def test_lazy_package_discovery_finds_a_module_getattr_and_nothing_else(tmp_path):
    _write_tree(tmp_path, {
        "puremacro/__init__.py": "",
        "puremacro/lazy/__init__.py": "def __getattr__(name): ...\n",
        "puremacro/eager/__init__.py": "x = 1\n",
        "puremacro/eager/sub/__init__.py": "class C:\n    def __getattr__(self, name): ...\n",
        "puremacro/broken/__init__.py": "def (:\n",
    })
    assert _lazy_packages(tmp_path) == ["puremacro.lazy"]


def test_the_lazy_package_discovery_agrees_with_a_plain_text_search():
    """An independent control: were the analysis to find nothing by mistake, the
    backstop would run on nothing and pass."""
    inits = sorted((REPO / "puremacro").rglob("__init__.py"))
    assert len(inits) > 30
    expected = [".".join(i.relative_to(REPO).parent.parts) for i in inits
                if re.search(r"^def __getattr__\(", i.read_text(encoding="utf-8"), re.M)]
    assert _LAZY_PACKAGES == expected


@pytest.mark.parametrize("package", _LAZY_PACKAGES)
def test_lazy_package_all_names_resolve_at_runtime(package, monkeypatch):
    """The backstop for the resolver's trust in a lazy package: import it and read
    every name in its runtime ``__all__``. A name the package cannot serve fails,
    and a module missing only for an optional dependency is reported as a skip."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        if not _is_optional_dependency(exc):
            raise
        pytest.skip(f"{package} needs {exc.name}, which is not installed")
    assert Path(module.__file__).resolve().is_relative_to(REPO), (
        f"{package} was imported from {module.__file__}, not from this checkout")
    failures, tolerated = _lazy_all_problems(module)
    assert not failures, (f"{package} lists names it cannot serve:\n  " + "\n  ".join(failures))
    if tolerated:
        pytest.skip("optional dependencies are missing for: " + ", ".join(tolerated))


def test_every_internal_import_resolves():
    tracked = _tracked_paths(REPO)
    sources = sorted(p for p in tracked
                     if p.endswith(".py") and p.startswith(_IMPORT_SCAN_ROOTS)
                     and not p.startswith(_IMPORT_SCAN_SKIP))
    problems: list[str] = []
    scanned = checked = 0
    for rel in sources:
        try:
            source = (REPO / rel).read_bytes()
        except FileNotFoundError:
            continue                    # tracked, but deleted from the working tree mid-edit
        found, n = _check_imports(rel, source, tracked, REPO)
        problems += found
        checked += n
        scanned += 1
    assert scanned > 300 and checked > 500, (
        f"the scan is vacuous: {scanned} files, {checked} imports checked")
    # positive control for the name rule: the real package __init__ was read
    provided = _init_names(REPO, "puremacro/__init__.py")
    assert provided is not None and "__version__" in provided.bound, "package __init__ files were not read"
    assert not problems, (
        f"{len(problems)} import(s) name a module or a name that is not there:\n  "
        + "\n  ".join(problems))
