import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .helpers import SR, soundtrack, warp, write_audio


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_sync(self, reference, target, output, *options):
        return subprocess.run([sys.executable, str(ROOT / "start_sync.py"), "--reference", str(reference),
                               "--target", str(target), "--output", str(output), "--step", "2", *options],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)

    def test_full_pipeline_with_graphs_and_actual_output_validation(self):
        with tempfile.TemporaryDirectory(prefix="wavesync_русский ") as td:
            directory = Path(td)
            reference = soundtrack(14.)
            n = round(.010 * SR)
            target = np.r_[np.zeros(n), reference[:-n]]
            rp, tp = directory / "эталон.wav", directory / "русский звук.wav"
            write_audio(rp, reference)
            write_audio(tp, target)
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (rp, tp)]
            out = directory / "result"
            result = self.run_sync(rp, tp, out, "--refine-only")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((out / "REPORT.json").read_text())
            self.assertEqual(report["status"], "VERIFIED_PCM")
            self.assertTrue((out / "SYNCED_TARGET.wav").exists())
            self.assertGreaterEqual(len(report["output_pcm_validation"]), 5)
            errors = [abs(r["residual_sec"]) for r in report["output_pcm_validation"] if r["status"] == "VERIFIED"]
            self.assertLess(max(errors), 1 / SR)
            self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in (rp, tp)])
            self.assertGreaterEqual(len(list((out / "graphs").glob("*.jpg"))), 7)
            self.assertTrue(list((out / "graphs/micro_overlays").glob("*.jpg")))
            self.assertFalse((out / "work").exists())
            # Existing result is not silently overwritten by another run.
            second = self.run_sync(rp, tp, out, "--refine-only", "--no-plots")
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(report, json.loads((out / "REPORT.json").read_text()))

    def test_already_synced_track_is_kept_without_render(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            x = soundtrack(10.)
            write_audio(d / "ref.wav", x)
            write_audio(d / "target.wav", x)
            result = self.run_sync(d / "ref.wav", d / "target.wav", d / "out", "--refine-only", "--no-plots")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((d / "out/REPORT.json").read_text())
            self.assertEqual(report["status"], "NO_CONFIRMED_IMPROVEMENT")
            self.assertFalse(list((d / "out").glob("*.wav")))

    def test_final_validator_rejects_deliberately_shifted_render(self):
        from wavesync.config import Settings
        from wavesync.guard import render_verdict, verify_render
        from wavesync.model import TimeMap
        x = soundtrack(14)
        n = round(.007 * SR)
        bad = np.r_[np.zeros(n), x[:-n]]
        identity = TimeMap.affine(14)
        rows = verify_render(x, x, bad, identity, identity, Settings(anchor_step=2))
        self.assertEqual(render_verdict(rows), "REJECTED")

    def test_automatic_pipeline_corrects_a_large_initial_delay(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            x = soundtrack(18.)
            n = round(.275 * SR)
            y = np.r_[np.zeros(n), x[:-n]]
            write_audio(d / "ref.wav", x)
            write_audio(d / "target.wav", y)
            result = self.run_sync(d / "ref.wav", d / "target.wav", d / "out", "--no-plots")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((d / "out/REPORT.json").read_text())
            self.assertEqual(report["status"], "VERIFIED_PCM")
            verified = [r for r in report["output_pcm_validation"] if r["status"] == "VERIFIED"]
            self.assertLess(max(abs(r["residual_sec"]) for r in verified), 1 / SR)

    def test_unrelated_audio_produces_no_synced_file(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_audio(d / "ref.wav", soundtrack(16., 15))
            write_audio(d / "target.wav", soundtrack(16., 113))
            result = self.run_sync(d / "ref.wav", d / "target.wav", d / "out", "--no-plots")
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            report = json.loads((d / "out/REPORT.json").read_text())
            self.assertEqual(report["status"], "BLOCKED")
            self.assertFalse(list((d / "out").glob("*.wav")))

    def test_adaptive_pipeline_refines_a_nonlinear_local_error(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            x = soundtrack(24.)
            times = np.arange(len(x)) / SR
            correction = -.010 * np.exp(-((times - 12.) / 5.) ** 2)
            target = warp(x, (times + correction) * SR)
            write_audio(d / "ref.wav", x)
            write_audio(d / "target.wav", target)
            result = self.run_sync(d / "ref.wav", d / "target.wav", d / "out", "--refine-only", "--step", "5", "--no-plots")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((d / "out/REPORT.json").read_text())
            self.assertEqual(report["status"], "VERIFIED_PCM")
            self.assertGreater(len(report["adaptive_steps_sec"]), 1)
            errors = [abs(r["residual_sec"]) for r in report["output_pcm_validation"] if r["status"] == "VERIFIED"]
            self.assertLess(np.median(errors), 1 / SR)

    def test_explicit_edit_map_pipeline_is_validated(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            x = soundtrack(40., seed=55)
            filler = soundtrack(3., seed=156)
            target = np.r_[x[:15 * SR], filler, x[15 * SR:]]
            write_audio(d / "ref.wav", x)
            write_audio(d / "target.wav", target)
            edit = {"pieces": [{"target": [0., 15.], "reference": [0., 15.]},
                               {"target": [18., 43.], "reference": [15., 40.]}]}
            (d / "edit.json").write_text(json.dumps(edit))
            result = self.run_sync(d / "ref.wav", d / "target.wav", d / "out", "--map-json", str(d / "edit.json"), "--no-plots")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((d / "out/REPORT.json").read_text())
            self.assertEqual(report["status"], "VERIFIED_PCM")
            self.assertAlmostEqual(report["edit_gaps"][0]["target_gap_sec"], 3.)

    def test_loss_of_verified_pcm_in_one_region_blocks_output(self):
        from wavesync.config import Settings
        from wavesync.guard import render_verdict, verify_render
        from wavesync.model import TimeMap
        x = soundtrack(14.)
        corrupted = x.copy()
        corrupted[4 * SR:7 * SR] = 0.
        identity = TimeMap.affine(14.)
        rows = verify_render(x, x, corrupted, identity, identity, Settings(anchor_step=2))
        self.assertTrue(any(r["status"] == "REGRESSION" for r in rows))
        self.assertEqual(render_verdict(rows), "REJECTED")

    def test_training_and_holdout_pcm_windows_are_disjoint_at_all_steps(self):
        from wavesync.config import Settings
        from wavesync.guard import holdout_times
        from wavesync.micro import grid
        from wavesync.model import TimeMap
        identity = TimeMap.affine(24.)
        for step in (.5, 1., 2., 5.):
            settings = Settings(anchor_step=step)
            length = min(settings.micro_window, step / 8)
            span = min(.45, step / 6)
            train = [(t + delta - length / 2, t + delta + length / 2)
                     for t in grid(identity, settings) for delta in (-span, 0., span)]
            normal = set(grid(identity, settings, phase=.5))
            for t in holdout_times(identity, settings):
                offsets = (-span, 0., span) if t in normal else (0.,)
                for delta in offsets:
                    left, right = t + delta - length / 2, t + delta + length / 2
                    self.assertFalse(any(left < b and a < right for a, b in train), (step, t, delta))

    def test_plot_failure_does_not_mislabel_verified_audio(self):
        from wavesync.audio import probe
        from wavesync.cli import parser, synchronize
        from wavesync.config import Settings
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            x = soundtrack(10.)
            n = round(.006 * SR)
            write_audio(d / "ref.wav", x)
            write_audio(d / "target.wav", np.r_[np.zeros(n), x[:-n]])
            args = parser().parse_args(["--refine-only", "--output", str(d / "out")])
            with patch("wavesync.cli.plots", side_effect=RuntimeError("test plot failure")):
                report, out = synchronize(probe(d / "ref.wav"), probe(d / "target.wav"), args, Settings(anchor_step=2), lambda *a: None)
            self.assertEqual(report["status"], "VERIFIED_PCM")
            self.assertIn("plot_error", report)
            self.assertTrue((out / "SYNCED_TARGET.wav").exists())

    def test_cancel_after_render_removes_new_output_and_preserves_inputs(self):
        from wavesync.audio import probe
        from wavesync.cli import parser, synchronize
        from wavesync.config import Settings
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            x = soundtrack(10.)
            n = round(.006 * SR)
            write_audio(d / "ref.wav", x)
            write_audio(d / "target.wav", np.r_[np.zeros(n), x[:-n]])
            original_hash = hashlib.sha256((d / "target.wav").read_bytes()).hexdigest()
            args = parser().parse_args(["--refine-only", "--output", str(d / "out")])
            with patch("wavesync.cli.plots", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    synchronize(probe(d / "ref.wav"), probe(d / "target.wav"), args, Settings(anchor_step=2), lambda *a: None)
            report = json.loads((d / "out/REPORT.json").read_text())
            self.assertEqual(report["status"], "BLOCKED")
            self.assertIsNone(report["output"])
            self.assertFalse(list((d / "out").glob("*.wav")))
            self.assertEqual(hashlib.sha256((d / "target.wav").read_bytes()).hexdigest(), original_hash)

    def test_actual_pal_rate_render_is_synchronized(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            x = soundtrack(18., seed=44)
            rate = 25 / 24
            target_times = np.arange(round(len(x) / rate)) / SR
            target = warp(x, rate * target_times * SR)
            write_audio(d / "ref.wav", x)
            write_audio(d / "target.wav", target)
            result = self.run_sync(d / "ref.wav", d / "target.wav", d / "out", "--no-plots")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((d / "out/REPORT.json").read_text())
            self.assertEqual(report["status"], "VERIFIED_PCM")
            errors = [abs(r["residual_sec"]) for r in report["output_pcm_validation"] if r["status"] == "VERIFIED"]
            self.assertGreater(len(errors), 8)
            self.assertLess(np.median(errors), 1 / SR)


if __name__ == "__main__":
    unittest.main()
