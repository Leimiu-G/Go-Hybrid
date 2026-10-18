"""Regression tests for fail-closed CLI argument validation (no network)."""
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ENTRY = Path(__file__).with_name("job.py")


class ActionsDispatchArgumentTests(unittest.TestCase):
    def call_dispatch(self, *args):
        return subprocess.run(
            [sys.executable, str(ENTRY), "actions", "dispatch", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_missing_parallelism_reports_usage_not_traceback(self):
        result = self.call_dispatch("dummy.yml", "0123456", "demo", "3")
        self.assertEqual(result.returncode, 2)
        self.assertIn("<maxparallel>", result.stdout)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_invalid_parallelism_reports_error_not_traceback(self):
        result = self.call_dispatch("dummy.yml", "0123456", "demo", "3", "bad")
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integers", result.stderr)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_nonpositive_values_are_rejected(self):
        for shards, parallel in [("0", "2"), ("3", "-1"), ("-2", "1")]:
            with self.subTest(shards=shards, parallel=parallel):
                result = self.call_dispatch("dummy.yml", "0123456", "demo", shards, parallel)
                self.assertEqual(result.returncode, 2)
                self.assertIn("positive integers", result.stderr)
                self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_invalid_shards_reports_error_not_traceback(self):
        result = self.call_dispatch("dummy.yml", "0123456", "demo", "bad", "2")
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integers", result.stderr)
        self.assertNotIn("Traceback", result.stdout + result.stderr)



class ActionsArtifactPaginationTests(unittest.TestCase):
    """Exercise complete ZIP retrieval against the documented page defaults."""
    def exercise_fetch(self, count, failing_page=None, failing_download=False):
        import contextlib
        import importlib.util
        import io
        import tempfile
        import zipfile
        from urllib.parse import parse_qs, urlparse
        from unittest.mock import patch

        spec = importlib.util.spec_from_file_location("job_pagination_fixture", ENTRY)
        job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job)
        api_pages, downloaded = [], []
        artifacts, bodies = [], {}
        for index in range(count):
            body = io.BytesIO()
            with zipfile.ZipFile(body, "w") as archive:
                archive.writestr("evidence.txt", "artifact-%d" % index)
            bodies[index] = body.getvalue()
            artifacts.append({"id": index, "name": "shard-%03d" % index,
                              "size_in_bytes": len(bodies[index]),
                              "archive_download_url": "https://artifacts.invalid/%d" % index})

        def api(path):
            query = parse_qs(urlparse(path).query)
            size = int(query.get("per_page", ["30"])[0])
            page = int(query.get("page", ["1"])[0])
            api_pages.append(page)
            if page == failing_page:
                raise RuntimeError("synthetic listing failure")
            start = (page - 1) * size
            return {"total_count": count, "artifacts": artifacts[start:start + size]}

        def download(req, timeout):
            self.assertEqual(timeout, 120)
            if failing_download:
                raise OSError("synthetic download failure")
            index = int(urlparse(req.full_url).path.rsplit("/", 1)[1])
            downloaded.append(index)
            return io.BytesIO(bodies[index])

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(job, "gh_api", side_effect=api), \
                    patch.object(job, "gh_token", return_value="synthetic-fixture"), \
                    patch.object(job.urllib.request, "urlopen", side_effect=download), \
                    contextlib.redirect_stdout(io.StringIO()):
                if failing_page is not None:
                    with self.assertRaisesRegex(RuntimeError, "listing failure"):
                        job.actions_fetch(123, directory)
                    self.assertEqual(downloaded, [])
                    self.assertFalse((Path(directory) / "result").exists())
                    return
                if failing_download:
                    with self.assertRaisesRegex(OSError, "download failure"):
                        job.actions_fetch(123, directory)
                    return
                job.actions_fetch(123, directory)
            files = sorted((Path(directory) / "result").glob("*.zip"))
            self.assertEqual(len(files), count)
            self.assertEqual(downloaded, list(range(count)))
            for index, file in enumerate(files):
                self.assertEqual(file.read_bytes(), bodies[index])
                with zipfile.ZipFile(file) as archive:
                    self.assertEqual(archive.read("evidence.txt").decode(), "artifact-%d" % index)
            self.assertEqual(api_pages, list(range(1, count // 100 + 2)))

    def test_retrieves_more_than_default_thirty_artifacts(self):
        self.exercise_fetch(31)

    def test_retrieves_later_pages(self):
        self.exercise_fetch(101)

    def test_empty_small_and_exact_page_boundaries(self):
        for count in (0, 1, 30, 100, 200):
            with self.subTest(count=count):
                self.exercise_fetch(count)

    def test_later_page_error_does_not_start_partial_download(self):
        self.exercise_fetch(101, failing_page=2)

    def test_download_errors_still_propagate(self):
        self.exercise_fetch(1, failing_download=True)


class KaggleFetchFreshnessTests(unittest.TestCase):
    """kaggle_fetch must promote only this invocation's summary (#478).

    A successful download can omit paths left by an older run, so the
    persistent cache mixes fresh and stale summaries; lexical order is not
    age (zzz-old sorts last). The fix stages the download and selects only
    from it, keeping every prior download in the cache.
    """

    IDENTITY = {"job": "demo", "sha": "a" * 40}

    def make_job(self, tmp):
        root = Path(tmp)
        raw = root / "result" / "raw"
        (raw / "zzz-old").mkdir(parents=True)
        stale = dict(self.IDENTITY, ok=True)
        (raw / "zzz-old" / "summary.json").write_text(
            json.dumps(stale), encoding="utf-8")
        return root, stale

    def run_fetch(self, root, download=None, rc=0):
        fetch = runpy.run_path(str(ENTRY))["kaggle_fetch"]

        def fake_sh(command, **kwargs):
            if download is not None:
                # The staged directory is the one .staging-* entry the fix
                # just created under the cache; read it from the
                # filesystem instead of parsing the command (quoting and
                # platform separators make parsing fragile).
                staging = next((root / "result" / "raw").glob(".staging-*"))
                download(staging)
            return rc, "offline download"

        with patch.dict(fetch.__globals__, {"sh": fake_sh}):
            return fetch(str(root), slug="fixture/no-network")

    def test_fresh_summary_wins_over_stale_zzz_old_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_job(tmp)
            fresh = dict(self.IDENTITY, ok=False)

            def download(staging):
                (staging / "summary.json").write_text(
                    json.dumps(fresh), encoding="utf-8")

            self.run_fetch(root, download=download)
            promoted = json.loads(
                (root / "result" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(promoted, fresh)

    def test_failed_download_does_not_promote_cached_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_job(tmp)
            with patch("sys.stdout", new=io.StringIO()) as out:
                returned = self.run_fetch(root, rc=1)
            self.assertEqual(returned, 1)
            self.assertIn("no summary promoted", out.getvalue())
            self.assertFalse((root / "result" / "summary.json").exists())
            # A failed run still retains its partial output for diagnosis.
            self.assertTrue(
                (root / "result" / "raw" / "zzz-old" / "summary.json").exists())

    def test_cache_retains_previous_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_job(tmp)
            fresh = dict(self.IDENTITY, ok=False)

            def download(staging):
                (staging / "summary.json").write_text(
                    json.dumps(fresh), encoding="utf-8")

            self.run_fetch(root, download=download)
            raw = root / "result" / "raw"
            self.assertTrue((raw / "zzz-old" / "summary.json").exists())
            self.assertEqual(
                json.loads((raw / "summary.json").read_text(encoding="utf-8")),
                fresh)

    def test_staging_is_cleaned_up(self):
        for rc in (0, 1):
            with self.subTest(rc=rc):
                with tempfile.TemporaryDirectory() as tmp:
                    root, _ = self.make_job(tmp)
                    fresh = dict(self.IDENTITY, ok=False)

                    def download(staging):
                        (staging / "summary.json").write_text(
                            json.dumps(fresh), encoding="utf-8")

                    self.run_fetch(root, download=download, rc=rc)
                    self.assertEqual(
                        list((root / "result" / "raw").glob(".staging-*")), [])

    def test_staging_lives_inside_the_job_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_job(tmp)
            captured = []

            def download(staging):
                captured.append(staging)

            self.run_fetch(root, download=download)
            self.assertEqual(len(captured), 1)
            raw = root / "result" / "raw"
            self.assertEqual(captured[0].resolve().parent, raw.resolve())

    def test_multiple_fresh_summaries_keep_lexical_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_job(tmp)
            first = dict(self.IDENTITY, ok=False, note="first")
            last = dict(self.IDENTITY, ok=False, note="last")

            def download(staging):
                for name, payload in [("aaa", first), ("bbb", last)]:
                    (staging / name).mkdir()
                    (staging / name / "summary.json").write_text(
                        json.dumps(payload), encoding="utf-8")

            self.run_fetch(root, download=download)
            promoted = json.loads(
                (root / "result" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(promoted, last)
            # The cache keeps the nested fresh artifacts too.
            raw = root / "result" / "raw"
            self.assertEqual(
                json.loads((raw / "aaa" / "summary.json").read_text(
                    encoding="utf-8")), first)
            self.assertEqual(
                json.loads((raw / "bbb" / "summary.json").read_text(
                    encoding="utf-8")), last)

    def test_missing_summary_warns_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_job(tmp)

            def download(staging):
                (staging / "kernel.log").write_text("no summary", encoding="utf-8")

            with patch("sys.stdout", new=io.StringIO()) as out:
                self.run_fetch(root, download=download)
            self.assertIn("no summary.json", out.getvalue())
            self.assertNotIn("Traceback", out.getvalue())
            self.assertFalse((root / "result" / "summary.json").exists())

    def test_uncacheable_summary_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_job(tmp)
            fresh = dict(self.IDENTITY, ok=False)

            def download(staging):
                (staging / "summary.json").write_text(
                    json.dumps(fresh), encoding="utf-8")
                (staging / "kernel.log").write_text("log", encoding="utf-8")

            real_copy2 = shutil.copy2

            def failing_copy2(src, dst, **kwargs):
                if Path(src).name == "summary.json":
                    raise OSError("disk full")
                return real_copy2(src, dst, **kwargs)

            fetch = runpy.run_path(str(ENTRY))["kaggle_fetch"]
            with patch("sys.stdout", new=io.StringIO()) as out, \
                    patch.dict(fetch.__globals__, {
                        "sh": self._fake_sh(root, download, 0),
                        "shutil": SimpleNamespace(
                            copy2=failing_copy2, rmtree=shutil.rmtree)}):
                fetch(str(root), slug="fixture/no-network")
            self.assertIn("could not cache summary.json", out.getvalue())
            self.assertFalse((root / "result" / "summary.json").exists())
            self.assertTrue((root / "result" / "raw" / "kernel.log").exists())
            self.assertEqual(
                list((root / "result" / "raw").glob(".staging-*")), [])

    @staticmethod
    def _fake_sh(root, download, rc):
        def fake_sh(command, **kwargs):
            if download is not None:
                staging = next((root / "result" / "raw").glob(".staging-*"))
                download(staging)
            return rc, "offline download"
        return fake_sh

    @unittest.skipUnless(os.name == "posix", "job.py's sh() runs /bin/sh")
    @unittest.skipUnless(
        (Path(ENTRY).parent.parent / "worker" / "kernel-metadata.json").exists(),
        "worker/kernel-metadata.json provides the kernel slug")
    def test_cli_failed_download_exits_nonzero(self):
        # End to end: kaggle_fetch returns the exit status and main()
        # propagates it, so `job.py kaggle fetch` fails closed at the
        # shell instead of exiting 0 after a failed download.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "result" / "raw").mkdir(parents=True)
            env = dict(os.environ, KAGGLE_CLI="/nonexistent/kaggle")
            result = subprocess.run(
                [sys.executable, str(ENTRY), "kaggle", "fetch", str(root)],
                capture_output=True, text=True, timeout=30, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no summary promoted", result.stdout)
            self.assertNotIn("Traceback", result.stdout + result.stderr)
            self.assertEqual(
                list((root / "result" / "raw").glob(".staging-*")), [])


if __name__ == "__main__":
    unittest.main()
