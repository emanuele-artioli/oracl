#!/usr/bin/env python3
"""
ORACL Simulation – end-to-end adaptive-streaming simulation with
residual-correction rounds, real ffmpeg encoding, and PSNR measurement.

Usage:
    python oracl_sim.py                        # uses default input.mp4
    python oracl_sim.py --input path/to/video.mp4
    python oracl_sim.py --input video.mp4 --cleanup
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

RESOLUTIONS = {
    "360p":  (640,  360),
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
}
QUALITY_LADDER = ["360p", "720p", "1080p", "1440p"]   # ordered low→high

CHUNK_DURATION = 4          # seconds
HEVC_CRF       = 28        # constant-rate-factor for x265 base encoding
RESIDUAL_CRF   = 18        # lower CRF for residuals to preserve detail
BUFFER_TARGET  = 8.0        # seconds – ABR setpoint
HW_ACCEL       = 'auto'     # 'auto'|'nvenc'|'cpu' — choose hardware encoder
WORKERS        = None       # None -> os.cpu_count(), used where parallelization is safe

# Network trace: list of (start_time_s, downlink_Mbps, uplink_Mbps)
# The bandwidth active at global_time is the last entry whose start_time <= global_time.
NETWORK_TRACE = [
    (0,   5.0,  1.5),
    (10,  3.0,  1.0),
    (20,  8.0,  2.5),
    (30,  4.0,  1.2),
    (40,  6.0,  2.0),
    (50, 10.0,  3.0),
    (60,  2.0,  0.8),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("ERROR: ffmpeg not found on PATH. Please install ffmpeg.")


def run_ffmpeg(args, desc="ffmpeg"):
    """Run an ffmpeg command, raising on failure."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed:\n{result.stderr}")


def detect_hw_accel(preferred='auto'):
    """Detect available hardware encoder. Returns 'nvenc' or 'cpu'."""
    if preferred == 'nvenc':
        return 'nvenc'
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        encs = out.stdout
        if 'hevc_nvenc' in encs or 'h264_nvenc' in encs:
            return 'nvenc'
    except Exception:
        pass
    return 'cpu'


def probe_duration(path):
    """Return duration of a media file in seconds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def probe_resolution(path):
    """Return (width, height) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", path],
        capture_output=True, text=True, check=True,
    )
    w, h = out.stdout.strip().split("x")
    return int(w), int(h)


def file_size_bits(path):
    return os.path.getsize(path) * 8


def bandwidth_at(global_time):
    """Return (downlink_bps, uplink_bps) from the network trace."""
    dl, ul = NETWORK_TRACE[0][1], NETWORK_TRACE[0][2]
    for t, d, u in NETWORK_TRACE:
        if t <= global_time:
            dl, ul = d, u
        else:
            break
    return dl * 1e6, ul * 1e6


# ── 1. Pre-Processing ───────────────────────────────────────────────────────

def generate_bitrate_ladder(input_path, output_dir):
    """
    Encode the input video at each resolution in the ladder and split
    into independent 4-second chunks.
    Returns: (num_chunks, ground_truth_resolution)
    """
    duration = probe_duration(input_path)
    gt_w, gt_h = probe_resolution(input_path)
    num_chunks = math.ceil(duration / CHUNK_DURATION)

    print(f"Input : {input_path}  ({gt_w}x{gt_h}, {duration:.2f}s, {num_chunks} chunks)")

    # If outputs already exist and have expected number of chunks, skip re-encoding
    expected_chunks = math.ceil(duration / CHUNK_DURATION)
    all_exist = True
    for label, (w, h) in RESOLUTIONS.items():
        res_dir = os.path.join(output_dir, label)
        if not os.path.isdir(res_dir) or len([f for f in os.listdir(res_dir) if f.startswith('chunk_')]) < expected_chunks:
            all_exist = False
            break
    gt_dir = os.path.join(output_dir, "ground_truth")
    if all_exist and os.path.isdir(gt_dir) and len([f for f in os.listdir(gt_dir) if f.startswith('chunk_')]) >= expected_chunks:
        print("Encoded ladder and ground-truth already exist — skipping encodes.")
    else:
        hw = detect_hw_accel(HW_ACCEL)
        if hw == 'nvenc':
            # Single-pass multi-output using ffmpeg split -> scale -> encode (NVENC)
            n = len(RESOLUTIONS)
            # Create split outputs labels [v0][v1]...
            split_labels = ''.join(f'[v{i}]' for i in range(n))
            filters = [f"[0:v]split={n}{split_labels}"]
            # Add scale filters that write to [v0s],[v1s],...
            for i, (label, (w, h)) in enumerate(RESOLUTIONS.items()):
                filters.append(f"[v{i}]scale={w}:{h}[v{i}s]")
            # Ensure directories exist
            for label in RESOLUTIONS.keys():
                os.makedirs(os.path.join(output_dir, label), exist_ok=True)
            filter_complex = ";".join(filters)
            # Build args: map [v0s] -> output0, [v1s] -> output1, ...
            args = ["-i", input_path, "-filter_complex", filter_complex]
            for i, label in enumerate(RESOLUTIONS.keys()):
                out_full = os.path.join(output_dir, label, "full.mp4")
                args += ["-map", f"[v{i}s]", "-c:v", "hevc_nvenc", "-preset", "p6", "-rc", "vbr_hq", "-cq", str(max(18, HEVC_CRF)), "-an", out_full]
            run_ffmpeg(args, desc="encode ladder (nvenc)")
            # Segment each full.mp4
            for label in RESOLUTIONS.keys():
                res_dir = os.path.join(output_dir, label)
                full_path = os.path.join(res_dir, "full.mp4")
                chunk_pattern = os.path.join(res_dir, "chunk_%d.mp4")
                run_ffmpeg([
                    "-i", full_path,
                    "-c", "copy",
                    "-f", "segment",
                    "-segment_time", str(CHUNK_DURATION),
                    "-reset_timestamps", "1",
                    chunk_pattern,
                ], desc=f"segment {label}")
                os.remove(full_path)
                print(f"  {label:>5s} : {RESOLUTIONS[label][0]}x{RESOLUTIONS[label][1]}  – {len(os.listdir(res_dir))} chunks")
        else:
            # Fallback: original CPU-based encode per-resolution
            for label, (w, h) in RESOLUTIONS.items():
                res_dir = os.path.join(output_dir, label)
                os.makedirs(res_dir, exist_ok=True)

                # Full-resolution encode first
                full_path = os.path.join(res_dir, "full.mp4")
                run_ffmpeg([
                    "-i", input_path,
                    "-vf", f"scale={w}:{h}",
                    "-c:v", "libx265", "-crf", str(HEVC_CRF),
                    "-preset", "fast",
                    "-an",        # drop audio for the simulation
                    full_path,
                ], desc=f"encode {label}")

                # Segment into chunks using the segment muxer
                chunk_pattern = os.path.join(res_dir, "chunk_%d.mp4")
                run_ffmpeg([
                    "-i", full_path,
                    "-c", "copy",
                    "-f", "segment",
                    "-segment_time", str(CHUNK_DURATION),
                    "-reset_timestamps", "1",
                    chunk_pattern,
                ], desc=f"segment {label}")

                os.remove(full_path)
                print(f"  {label:>5s} : {w}x{h}  – {len(os.listdir(res_dir))} chunks")

    # Also produce ground-truth chunks at the original resolution
    gt_dir = os.path.join(output_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)
    gt_full = os.path.join(gt_dir, "full.mp4")
    # Use high-quality CRF but not lossless to save time; set to RESIDUAL_CRF for balance
    run_ffmpeg([
        "-i", input_path,
        "-c:v", "libx264", "-crf", str(max(18, RESIDUAL_CRF)), "-preset", "veryfast", "-an",
        gt_full,
    ], desc="encode ground-truth (high-quality)")
    chunk_pattern = os.path.join(gt_dir, "chunk_%d.mp4")
    run_ffmpeg([
        "-i", gt_full,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(CHUNK_DURATION),
        "-reset_timestamps", "1",
        chunk_pattern,
    ], desc="segment ground-truth")
    os.remove(gt_full)
    print(f"  ground_truth : {gt_w}x{gt_h}  – {len(os.listdir(gt_dir))} chunks")

    return num_chunks, (gt_w, gt_h)


# ── Video-frame I/O helpers ──────────────────────────────────────────────────

def decode_chunk_frames(chunk_path):
    """Decode a chunk to a list of numpy uint8 frames (BGR)."""
    cap = cv2.VideoCapture(chunk_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {chunk_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def encode_frames_to_file(frames, out_path, fps=30.0, crf=None, hw=None):
    """
    Encode a list of uint8 BGR frames to a .mp4 file. Uses NVENC when hw=='nvenc'.
    Returns actual file size in bits.
    """
    if crf is None:
        crf = RESIDUAL_CRF
    h, w = frames[0].shape[:2]
    if hw is None:
        hw = detect_hw_accel(HW_ACCEL)

    # Choose encoder based on hw
    if hw == 'nvenc':
        encoder_args = ["-c:v", "hevc_nvenc", "-preset", "p6", "-rc", "vbr_hq", "-cq", str(max(18, crf))]
    else:
        encoder_args = ["-c:v", "libx265", "-crf", str(crf), "-preset", "fast"]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
    ] + encoder_args + ["-pix_fmt", "yuv420p", out_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"encode_frames_to_file failed:\n{proc.stderr.read().decode()}")
    return file_size_bits(out_path)


def get_chunk_fps(chunk_path):
    """Return the FPS of a chunk file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", chunk_path],
        capture_output=True, text=True, check=True,
    )
    num, den = out.stdout.strip().split("/")
    return float(num) / float(den)


# ── Residual arithmetic ─────────────────────────────────────────────────────

def compute_residual_frames(target_frames, source_frames):
    """
    Vectorized residual computation: target - source mapped to uint8 via +128 offset.
    Returns list of uint8 frames.
    """
    # Stack for vectorized ops
    ta = np.stack(target_frames).astype(np.int16)
    sa = np.stack(source_frames).astype(np.int16)
    diff = ta - sa   # signed
    mapped = np.clip(diff + 128, 0, 255).astype(np.uint8)
    return [mapped[i] for i in range(mapped.shape[0])]


def apply_residual_frames(base_frames, residual_frames):
    """
    Vectorized inverse of compute_residual_frames: base + (residual - 128), clipped.
    """
    ba = np.stack(base_frames).astype(np.int16)
    ra = np.stack(residual_frames).astype(np.int16)
    corrected = ba + (ra - 128)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return [corrected[i] for i in range(corrected.shape[0])]


# ── PSNR ─────────────────────────────────────────────────────────────────────

def psnr_frames(frames_a, frames_b):
    """Average PSNR across corresponding frames using cv2.PSNR (fast C impl)."""
    vals = []
    for a, b in zip(frames_a, frames_b):
        vals.append(cv2.PSNR(a, b))
    return np.mean(vals)


# ── 2. Simulation Loop ──────────────────────────────────────────────────────

def simulate(output_dir, num_chunks, gt_res, cleanup):
    gt_w, gt_h = gt_res

    global_time      = 0.0
    buffer_level     = BUFFER_TARGET   # start with a full buffer
    quality_index    = 0               # start at lowest quality
    tmp_dir          = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    results = []

    print(f"\n{'='*80}")
    print(f"{'Chunk':>6} {'Requested':>10} {'Base kB':>9} {'R1 kB':>8} "
          f"{'R2 kB':>8} {'DL1 ms':>8} {'UL ms':>8} {'DL2 ms':>8} "
          f"{'PSNR dB':>9} {'Buffer s':>9}")
    print(f"{'='*80}")

    for chunk_idx in range(num_chunks):
        chunk_start_time = global_time

        # ── Step A: ABR decision ─────────────────────────────────────────
        if buffer_level < BUFFER_TARGET:
            quality_index = max(0, quality_index - 1)
        else:
            quality_index = min(len(QUALITY_LADDER) - 1, quality_index + 1)
        requested_quality = QUALITY_LADDER[quality_index]

        # -- Downlink 1: transfer requested base chunk --
        base_chunk_path = os.path.join(
            output_dir, requested_quality, f"chunk_{chunk_idx}.mp4")
        if not os.path.isfile(base_chunk_path):
            print(f"  [WARN] {base_chunk_path} missing – skipping chunk {chunk_idx}")
            continue

        base_size = file_size_bits(base_chunk_path)
        dl_bps, ul_bps = bandwidth_at(global_time)
        dl1_time = base_size / dl_bps
        global_time += dl1_time

        # ── Step B: Client upscaling + Residual_1 ────────────────────────
        base_frames = decode_chunk_frames(base_chunk_path)
        fps = get_chunk_fps(base_chunk_path)

        # Upscale base to ground-truth resolution (bicubic)
        upscaled_frames = [
            cv2.resize(f, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC)
            for f in base_frames
        ]

        # Ground-truth chunk
        gt_chunk_path = os.path.join(
            output_dir, "ground_truth", f"chunk_{chunk_idx}.mp4")
        gt_frames = decode_chunk_frames(gt_chunk_path)

        # Match frame counts (last chunk may differ slightly between encodes)
        min_len = min(len(upscaled_frames), len(gt_frames))
        upscaled_frames = upscaled_frames[:min_len]
        gt_frames       = gt_frames[:min_len]

        # Residual_1 = upscaled - gt  (from client's perspective, it sends
        # its "hallucinated" error to the server for correction)
        res1_frames = compute_residual_frames(upscaled_frames, gt_frames)
        res1_path   = os.path.join(tmp_dir, f"res1_chunk_{chunk_idx}.mp4")
        res1_bits   = encode_frames_to_file(res1_frames, res1_path, fps=fps)

        # ── Step C: Uplink – send Residual_1 to server ───────────────────
        _, ul_bps = bandwidth_at(global_time)
        ul_time = res1_bits / ul_bps
        global_time += ul_time

        # ── Step D: Server correction ────────────────────────────────────
        # Server decodes Residual_1
        decoded_res1 = decode_chunk_frames(res1_path)
        decoded_res1 = decoded_res1[:min_len]

        # Server reconstructs client view: base_upscaled + Residual_1
        client_view = apply_residual_frames(upscaled_frames, decoded_res1)

        # Correction residual: Residual_2 = GT - client_view
        res2_frames = compute_residual_frames(gt_frames, client_view)
        res2_path   = os.path.join(tmp_dir, f"res2_chunk_{chunk_idx}.mp4")
        res2_bits   = encode_frames_to_file(res2_frames, res2_path, fps=fps)

        # ── Step E: Downlink 2 – send Residual_2 to client ──────────────
        dl_bps, _ = bandwidth_at(global_time)
        dl2_time = res2_bits / dl_bps
        global_time += dl2_time

        # Client reconstructs final: upscaled + res1 + res2
        decoded_res2 = decode_chunk_frames(res2_path)
        decoded_res2 = decoded_res2[:min_len]

        final_frames = apply_residual_frames(client_view, decoded_res2)

        # PSNR: final vs ground-truth
        chunk_psnr = psnr_frames(final_frames, gt_frames)

        # Buffer update
        total_round_time = global_time - chunk_start_time
        buffer_level += CHUNK_DURATION - total_round_time
        buffer_level = max(buffer_level, 0.0)

        # Log
        base_kb = base_size / 8 / 1024
        r1_kb   = res1_bits / 8 / 1024
        r2_kb   = res2_bits / 8 / 1024

        print(f"{chunk_idx:>6d} {requested_quality:>10s} {base_kb:>9.1f} {r1_kb:>8.1f} "
              f"{r2_kb:>8.1f} {dl1_time*1000:>8.1f} {ul_time*1000:>8.1f} "
              f"{dl2_time*1000:>8.1f} {chunk_psnr:>9.2f} {buffer_level:>9.2f}")

        results.append({
            "chunk":    chunk_idx,
            "quality":  requested_quality,
            "base_kb":  base_kb,
            "r1_kb":    r1_kb,
            "r2_kb":    r2_kb,
            "dl1_ms":   dl1_time * 1000,
            "ul_ms":    ul_time * 1000,
            "dl2_ms":   dl2_time * 1000,
            "psnr":     chunk_psnr,
            "buffer":   buffer_level,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    if results:
        avg_psnr = np.mean([r["psnr"] for r in results if r["psnr"] != float("inf")])
        total_dl  = sum(r["base_kb"] + r["r2_kb"] for r in results)
        total_ul  = sum(r["r1_kb"] for r in results)
        print(f"\n{'─'*80}")
        print(f"  Average PSNR     : {avg_psnr:.2f} dB")
        print(f"  Total downlink   : {total_dl:.1f} kB")
        print(f"  Total uplink     : {total_ul:.1f} kB")
        print(f"  Final global time: {global_time:.3f} s")
        print(f"{'─'*80}")

    # Cleanup
    if cleanup:
        print("\nCleaning up temporary and chunk files...")
        shutil.rmtree(output_dir)
        print("Done.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ORACL adaptive-streaming simulation")
    parser.add_argument("--input", "-i", default="input.mp4",
                        help="Path to input video file (default: input.mp4)")
    parser.add_argument("--output-dir", "-o", default="output",
                        help="Directory for encoded chunks (default: output/)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete all generated files after the simulation")
    parser.add_argument("--hw-accel", choices=["auto", "nvenc", "cpu"], default='auto',
                        help="Hardware acceleration preference (default: auto)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes for parallel tasks (default: system cores)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: Input file '{args.input}' not found.")

    check_ffmpeg()

    # Configure globals
    global HW_ACCEL, WORKERS
    HW_ACCEL = args.hw_accel
    WORKERS = args.workers if args.workers is not None else os.cpu_count()

    print(f"Selected HW accel: {HW_ACCEL}  |  workers: {WORKERS}")

    output_dir = os.path.abspath(args.output_dir)
    num_chunks, gt_res = generate_bitrate_ladder(args.input, output_dir)
    simulate(output_dir, num_chunks, gt_res, args.cleanup)


if __name__ == "__main__":
    main()
