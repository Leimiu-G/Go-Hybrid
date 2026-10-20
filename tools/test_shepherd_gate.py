#!/usr/bin/env python3
"""test_shepherd_gate.py — offline self-tests for tools/shepherd.py.

No network, no subprocesses: only the pure functions that the workflows lean
on. If the kill-switch matrix, the dup planner, the conflict classifier, or
the probe parser drift, CI fails before any workflow can.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shepherd as sh  # noqa: E402


class GateMatrix(unittest.TestCase):
    """The kill switch is the safety floor for every workflow (plan §1.5)."""

    def test_off_blocks_everything(self):
        for req in ("read", "dry-run", "live"):
            self.assertFalse(sh.mode_allows("off", req), req)

    def test_dry_run_allows_read_and_comment_only(self):
        self.assertTrue(sh.mode_allows("dry-run", "read"))
        self.assertTrue(sh.mode_allows("dry-run", "dry-run"))
        self.assertFalse(sh.mode_allows("dry-run", "live"))

    def test_live_allows_all(self):
        for req in ("read", "dry-run", "live"):
            self.assertTrue(sh.mode_allows("live", req), req)

    def test_unknown_mode_falls_back_to_dry_run(self):
        self.assertTrue(sh.mode_allows("bogus", "read"))
        self.assertFalse(sh.mode_allows("bogus", "live"))


def _item(n, **kw):
    it = {"number": n, "wildcard_hold": False, "draft_parked": False,
          "dup_members": None, "dup_of": None}
    it.update(kw)
    return it


class DupPlanner(unittest.TestCase):
    """W4 Phase A must never propose against closed survivors, wildcard
    holds, or drafts (the [474, 616, 945] and [620, 910] lessons)."""

    def test_open_cluster_planned(self):
        items = [_item(5, dup_members=[6, 7]), _item(6), _item(7)]
        clusters = [[5, 6, 7]]
        plan = sh.plan_phase_a(items, clusters, is_open=lambda n: n in (5, 6, 7))
        self.assertEqual(plan, [(5, [6, 7])])

    def test_closed_survivor_skips_cluster(self):
        items = [_item(620, dup_members=[910]), _item(910)]
        clusters = [[620, 910]]
        plan = sh.plan_phase_a(items, clusters, is_open=lambda n: n == 910)
        self.assertEqual(plan, [])

    def test_wildcard_survivor_and_members_skipped(self):
        items = [_item(616, dup_members=[474, 945], wildcard_hold=True),
                 _item(474, wildcard_hold=True), _item(945)]
        clusters = [[474, 616, 945]]
        plan = sh.plan_phase_a(items, clusters, is_open=lambda n: True)
        self.assertEqual(plan, [])

    def test_wildcard_member_filtered_leaves_no_pairs(self):
        # cluster [384, 572]: survivor 572 open, member 384 wildcard-held ->
        # the only member is filtered out, so the cluster yields nothing.
        items = [_item(572, dup_members=[384]), _item(384, wildcard_hold=True)]
        clusters = [[384, 572]]
        plan = sh.plan_phase_a(items, clusters, is_open=lambda n: True)
        self.assertEqual(plan, [])


class ConflictClassifier(unittest.TestCase):
    """Docs-only conflicts resolve deterministically; anything else is human
    territory (the #870 policy, plan §3 W5)."""

    def test_docs_only(self):
        docs, real = sh.classify_conflicts(["PLAN.md", "STATUS.md"],
                                           {"PLAN.md", "STATUS.md"})
        self.assertEqual(docs, ["PLAN.md", "STATUS.md"])
        self.assertEqual(real, [])

    def test_any_code_conflict_is_real(self):
        docs, real = sh.classify_conflicts(["PLAN.md", "src/hycheck/buildable.rs"],
                                           {"PLAN.md", "STATUS.md"})
        self.assertEqual(docs, ["PLAN.md"])
        self.assertEqual(real, ["src/hycheck/buildable.rs"])


class LabelReconcile(unittest.TestCase):
    """`sync-labels` must be able to take a state label back off, not only put
    one on. `rebase.exempt_labels` contains `bot:conflict`, so a label left
    behind pins its PR out of W2 permanently: 8 of the 35 open PRs under
    `bot:conflict` report mergeable=true / mergeable_state=behind on
    2026-10-20, and the label predates the fix that cleared it (#986)."""

    MAPPING = {"LAND-CANDIDATE": "bot:land-candidate",
               "CONFLICTS": "bot:conflict",
               "STALE-BASE": "bot:stale-base"}

    def test_stale_conflict_is_removed_once_the_conflict_is_gone(self):
        add, remove = sh.label_plan("STALE-BASE",
                                    ["bot:conflict", "triage:new"],
                                    self.MAPPING)
        self.assertEqual(remove, ["bot:conflict"])
        self.assertEqual(add, ["bot:stale-base"])

    def test_current_state_costs_nothing(self):
        add, remove = sh.label_plan("CONFLICTS", ["bot:conflict"],
                                    self.MAPPING)
        self.assertEqual((add, remove), ([], []))

    def test_bucket_without_a_label_clears_every_state_label(self):
        add, remove = sh.label_plan("RECON",
                                    ["bot:land-candidate", "bot:stale-base"],
                                    self.MAPPING)
        self.assertEqual(add, [])
        self.assertEqual(remove, ["bot:land-candidate", "bot:stale-base"])

    def test_only_the_two_state_labels_coexist_never(self):
        add, remove = sh.label_plan("CONFLICTS",
                                    ["bot:stale-base", "bot:conflict"],
                                    self.MAPPING)
        self.assertEqual(add, [])
        self.assertEqual(remove, ["bot:stale-base"])

    def test_labels_owned_elsewhere_are_never_removed(self):
        add, remove = sh.label_plan("RECON",
                                    ["bot:ready-to-land", "bot:verified",
                                     "bot:landed", "bot:preflight-pass",
                                     "bot:stale-premise", "triage:keep"],
                                    self.MAPPING)
        self.assertEqual((add, remove), ([], []))

    def test_absent_labels_key_is_survivable(self):
        # items built by an older TRIAGE.json have no "labels" key at all;
        # the scan must still add the bucket label rather than crash.
        add, remove = sh.label_plan("STALE-BASE", None, self.MAPPING)
        self.assertEqual(add, ["bot:stale-base"])
        self.assertEqual(remove, [])


class SyncLabelCalls(unittest.TestCase):
    """Drive the real `sync-labels` command and inspect the HTTP verbs it
    emits. With the pre-#986 code the stale `bot:conflict` survives every run
    because the command only knows how to POST one on."""

    def _run(self, items):
        import argparse
        import json as _json
        import os
        import tempfile

        calls = []

        class FakeGh:
            repo = "o/r"
            calls = 0
            remaining = 999

            def __init__(self, *a, **k):
                pass

            def post(self, path, body=None):
                calls.append(("POST", path, body))
                return 200, {}

            def req(self, method, path, body=None):
                calls.append((method, path, body))
                return 200, None

        fd, scan = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            _json.dump({"items": items}, fh)
        args = argparse.Namespace(from_scan=scan, dry=False)
        orig, had = sh.tb.Gh, os.environ.get("GITHUB_TOKEN")
        os.environ["GITHUB_TOKEN"] = "unit-test-token"
        sh.tb.Gh = FakeGh
        try:
            sh.cmd_sync_labels(args)
        finally:
            sh.tb.Gh = orig
            if had is None:
                os.environ.pop("GITHUB_TOKEN", None)
            else:
                os.environ["GITHUB_TOKEN"] = had
            os.unlink(scan)
        # keep only the per-PR label mutations, not ensure_palette's chatter
        return [c for c in calls if "/issues/" in c[1]]

    def test_stale_conflict_is_deleted_not_just_overposted(self):
        # #711 on 2026-10-20: labelled bot:conflict at 05:44, force-pushed the
        # fix at 06:27, still carrying the label afterwards.
        out = self._run([{"number": 711, "bucket": "STALE-BASE",
                          "labels": ["bot:conflict", "triage:new"],
                          "wildcard_hold": False, "draft_parked": False}])
        verbs = [(v, p.split("/")[-1]) for v, p, _ in out]
        self.assertIn(("DELETE", "bot%3Aconflict"), verbs)
        self.assertIn(("POST", "labels"), verbs)

    def test_steady_state_costs_no_label_calls(self):
        # the whole point of reconciling from the scan payload rather than
        # fetching labels per PR: a PR already in sync must cost 0 calls.
        out = self._run([{"number": 819, "bucket": "CONFLICTS",
                          "labels": ["bot:conflict"],
                          "wildcard_hold": False, "draft_parked": False}])
        self.assertEqual(out, [])

    def test_a_genuinely_conflicted_pr_is_still_labelled(self):
        out = self._run([{"number": 931, "bucket": "CONFLICTS", "labels": [],
                          "wildcard_hold": False, "draft_parked": False}])
        verbs = [(v, p.split("/")[-1]) for v, p, _ in out]
        self.assertEqual(verbs, [("POST", "labels")])
        self.assertEqual(out[0][2], {"labels": ["bot:conflict"]})

    def test_parked_prs_stay_untouched(self):
        for key in ("wildcard_hold", "draft_parked"):
            out = self._run([{"number": 5, "bucket": "CONFLICTS",
                              "labels": ["bot:conflict"], key: True}])
            self.assertEqual(out, [], key)


class ProbeParser(unittest.TestCase):
    """W6 parses added diff lines only, and matches the #804 registry entry."""

    DIFF = ("diff --git a/x.hy b/x.hy\n"
            "--- a/x.hy\n"
            "+++ b/x.hy\n"
            "@@ -1,3 +1,4 @@\n"
            " keep me\n"
            "-removed line with make(<-chan int)\n"
            "+let c = make(<-chan int, 1)\n"
            "+++ not an added line\n")

    def test_added_lines_skip_context_removed_and_file_headers(self):
        # '+'-lines only: context, '-'-lines, and '+++' file headers are not
        # additions (a removed line proving a premise does not count).
        lines = list(sh.added_lines(self.DIFF))
        self.assertEqual(lines, ["let c = make(<-chan int, 1)"])

    def test_registry_matches_added_only(self):
        rule = sh.PROBES[0]
        added = list(sh.added_lines(self.DIFF))
        self.assertTrue(any(re.search(rule["pattern"], ln) for ln in added))
        self.assertFalse(any("removed line" in ln for ln in added))


if __name__ == "__main__":
    unittest.main()
