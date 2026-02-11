# ORACL creates a high-fidelity video stream by sending a low-resolution "base" chunk to the client, having the client upscale it, and then using the network to exchange video-encoded residuals to correct the errors.

import os
import shlex
import subprocess
import logging
import time
import math
import csv
import re
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup(input_file: str,
          output_dir: str = "output",
          resolutions: Optional[List[int]] = None,
          crfs: Optional[Dict[int, int]] = None,
          chunk_seconds: int = 4,
          audio_bitrate: str = "128k",
          preset: str = "medium",
          metrics_csv: Optional[str] = None) -> Dict[int, List[str]]:
    """Transcode `input_file` into a CRF-based HEVC ladder and split each resolution into
    independent 4-second mp4 chunks using libx265 (HEVC). Also generates a CSV with
    per-chunk statistics: chunk_name, file_size, encoding_time, ssim_score.

    Args:
        input_file: Path to input video (e.g. 'input.mp4').
        output_dir: Directory where per-resolution subdirs and chunks will be written.
        resolutions: List of vertical resolutions (heights). Defaults to [360, 720, 1080, 1440].
        crfs: Optional mapping from resolution height to CRF value (lower => higher quality).
              If omitted sensible defaults are used (lower CRF for larger resolutions).
        chunk_seconds: Segment length in seconds (default 4).
        audio_bitrate: Audio bitrate string for AAC audio.
        preset: x265 preset (e.g., 'slow', 'medium', 'fast').
        metrics_csv: Path to CSV file to write metrics. Defaults to <output_dir>/metrics.csv.

    Returns:
        A dict mapping resolution -> list of output chunk file paths.
    """

    if resolutions is None:
        resolutions = [360, 720, 1080, 1440]

    # sensible default CRF per resolution (lower CRF => higher quality)
    default_crfs = {
        360: 28,
        720: 23,
        1080: 20,
        1440: 18,
    }
    if crfs is None:
        crfs = default_crfs
    else:
        for r, v in default_crfs.items():
            crfs.setdefault(r, v)

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    os.makedirs(output_dir, exist_ok=True)

    if metrics_csv is None:
        metrics_csv = os.path.join(output_dir, "metrics.csv")

    # Prepare CSV file (write header if empty/non-existent)
    need_header = not os.path.exists(metrics_csv)
    csv_file = open(metrics_csv, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if need_header:
        csv_writer.writerow(["resolution", "chunk_name", "file_size", "encoding_time", "ssim_score"])
        csv_file.flush()

    # helper: probe duration
    def _probe_duration(path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        try:
            return float(out)
        except Exception:
            return 0.0

    # helper: probe original video height
    def _probe_height(path: str) -> Optional[int]:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=height",
            "-of", "csv=p=0",
            path,
        ]
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            return int(out)
        except Exception:
            return None

    # helper: compute SSIM between a segment of the original and a produced chunk at a given resolution
    def _compute_ssim(input_path: str, chunk_path: str, start: float, dur: float, height: int) -> Optional[float]:
        # Scale both streams to the target resolution and run ssim on the pair.
        cmd = [
            "ffmpeg", "-hide_banner",
            "-ss", str(start), "-i", input_path, "-t", str(dur),
            "-i", chunk_path,
            "-filter_complex",
            f"[0:v]scale=-2:{height}[ref];[1:v]scale=-2:{height}[dist];[ref][dist]ssim",
            "-f", "null", "-",
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            stderr = proc.stderr
            # Find last 'All:' occurrence and return the last one
            matches = re.findall(r"All:\s*([0-9]*\.?[0-9]+)", stderr)
            if matches:
                return float(matches[-1])
            # Fallback: look for more permissive patterns
            matches = re.findall(r"All[:=]\s*([0-9]*\.?[0-9]+)", stderr)
            if matches:
                return float(matches[-1])
            logger.debug("SSIM stderr did not contain an All: value; stderr: %s", stderr)
        except subprocess.CalledProcessError as e:
            logger.warning("SSIM computation failed for %s vs %s: %s", input_path, chunk_path, e.stderr)
        return None
    # total duration
    total_dur = _probe_duration(input_file)
    if total_dur <= 0:
        raise RuntimeError("Unable to determine input duration or duration is zero")

    results: Dict[int, List[str]] = {h: [] for h in resolutions}
    # key 0 will contain native/original chunks
    results[0] = []

    # probe original height for labeling and directory
    orig_height = _probe_height(input_file)
    original_dir = os.path.join(output_dir, "original")
    os.makedirs(original_dir, exist_ok=True)

    total_chunks = math.ceil(total_dur / chunk_seconds)
    logger.info("Encoding %d chunks of %ds each (last may be shorter) from %s", total_chunks, chunk_seconds, input_file)

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_seconds
        dur = min(chunk_seconds, max(0.0, total_dur - start))
        if dur <= 0:
            break

        # First: extract ground-truth chunk at native resolution (try stream copy, fallback to lossless re-encode)
        out_original = os.path.join(original_dir, f"chunk_{chunk_idx}.mp4")
        copy_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-i", input_file,
            "-t", str(dur),
            "-c:v", "copy", "-c:a", "copy",
            out_original,
        ]
        try:
            t0 = time.perf_counter()
            subprocess.run(copy_cmd, check=True, capture_output=True, text=True)
            t1 = time.perf_counter()
            copy_time = t1 - t0
            logger.info("Extracted original chunk %d (copy) -> %s", chunk_idx, out_original)
        except subprocess.CalledProcessError:
            # fallback: lossless re-encode to ensure independence and correctness
            reenc_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", input_file,
                "-t", str(dur),
                "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "lossless=1",
                "-c:a", "copy",
                out_original,
            ]
            logger.warning("Stream copy failed for original chunk %d, lossless re-encoding", chunk_idx)
            try:
                t0 = time.perf_counter()
                subprocess.run(reenc_cmd, check=True, capture_output=True, text=True)
                t1 = time.perf_counter()
                copy_time = t1 - t0
                logger.info("Re-encoded original chunk %d -> %s", chunk_idx, out_original)
            except subprocess.CalledProcessError as e:
                logger.error("Failed to extract or re-encode original chunk %d: %s", chunk_idx, e.stderr)
                copy_time = 0.0

        if os.path.exists(out_original):
            file_size = os.path.getsize(out_original)
            # ground-truth vs itself has SSIM 1.0
            ssim_val = 1.0
            rel_name = os.path.join("original", f"chunk_{chunk_idx}.mp4")
            csv_writer.writerow([str(orig_height) if orig_height is not None else "original", rel_name, str(file_size), f"{copy_time:.3f}", f"{ssim_val:.6f}"])
            csv_file.flush()
            results[0].append(out_original)

        for height in resolutions:
            res_dir = os.path.join(output_dir, f"{height}p")
            os.makedirs(res_dir, exist_ok=True)

            crf = crfs.get(height, default_crfs.get(height))
            out_file = os.path.join(res_dir, f"chunk_{chunk_idx}.mp4")
            vf = f"scale=-2:{height}"

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", input_file,
                "-t", str(dur),
                "-c:v", "libx265",
                "-preset", preset,
                "-vf", vf,
                "-crf", str(crf),
                "-c:a", "aac", "-b:a", audio_bitrate,
                "-force_key_frames", "expr:gte(t,0)",
                out_file,
            ]

            logger.info("Encoding chunk %d for %dp (start=%.1f dur=%.1f) -> %s", chunk_idx, height, start, dur, out_file)
            logger.debug("Command: %s", shlex.join(cmd))

            try:
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                t1 = time.perf_counter()
                encoding_time = t1 - t0
            except subprocess.CalledProcessError as e:
                logger.error("ffmpeg failed for chunk %d %dp: %s", chunk_idx, height, e.stderr)
                continue

            if not os.path.exists(out_file):
                logger.error("Expected output missing: %s", out_file)
                continue

            file_size = os.path.getsize(out_file)

            ssim = _compute_ssim(input_file, out_file, start, dur, height)

            rel_chunk_name = os.path.join(f"{height}p", f"chunk_{chunk_idx}.mp4")
            csv_writer.writerow([str(height), rel_chunk_name, str(file_size), f"{encoding_time:.3f}", "" if ssim is None else f"{ssim:.6f}"])
            csv_file.flush()

            results[height].append(out_file)
    csv_file.close()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CRF-based HEVC ladder and 4s mp4 chunks using libx265")
    parser.add_argument("input", help="Input video file (e.g., input.mp4)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--chunk-seconds", type=int, default=4, help="Chunk length in seconds")
    parser.add_argument("--crf", type=int, help="Global CRF to use for all resolutions (overrides defaults)")
    args = parser.parse_args()

    crfs = None
    if args.crf is not None:
        crfs = {360: args.crf, 720: args.crf, 1080: args.crf, 1440: args.crf}

    res = setup(args.input, output_dir=args.output, chunk_seconds=args.chunk_seconds, crfs=crfs)
    for h, chunks in res.items():
        print(f"{h}p: {len(chunks)} chunks -> first chunk: {chunks[0] if chunks else 'none'}")

