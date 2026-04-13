# ORACL creates a high-fidelity video stream by sending a low-resolution "base" chunk to the client, having the client upscale it, and then using the network to exchange video-encoded residuals to correct the errors.

import os
import subprocess
import logging
import time
import math
import csv
import re
import tempfile
import shutil
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple


# detect if ffmpeg has NVENC available
def _ffmpeg_has_encoder(name: str) -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        return name in proc.stdout
    except Exception:
        return False

NVENC_AVAILABLE = _ffmpeg_has_encoder("hevc_nvenc") or _ffmpeg_has_encoder("h264_nvenc")
ENCODER_HEVC = "hevc_nvenc" if _ffmpeg_has_encoder("hevc_nvenc") else "libx265"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)


# Detect ffmpeg/NVIDIA acceleration availability
def _ffmpeg_has_encoder(enc_name: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True)
        return enc_name in out
    except Exception:
        return False


def _ffmpeg_has_hwaccel(hw: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-hwaccels"], stderr=subprocess.STDOUT, text=True)
        return hw in out
    except Exception:
        return False


USE_NVENC = _ffmpeg_has_encoder("hevc_nvenc")
USE_HWACCEL_CUDA = _ffmpeg_has_hwaccel("cuda")
if USE_NVENC:
    logger.info("ffmpeg: NVENC available — GPU encoding enabled")
if USE_HWACCEL_CUDA:
    logger.info("ffmpeg: CUDA hwaccel available — will use hwaccel for ffmpeg operations")




# -------------------------- Shared helpers --------------------------

def decode_video_frames(path: str) -> Tuple[List, float, Tuple[int, int]]:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy required: install opencv-python and numpy")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, (w, h)


def encode_frames_to_hevc(frames: list, fps: float, out_path: str, crf: Optional[int] = None, preset: str = "medium") -> None:
    tmp = tempfile.mkdtemp(prefix="oracl_enc_")
    try:
        for i, f in enumerate(frames):
            fname = os.path.join(tmp, f"frame_{i:06d}.png")
            cv2.imwrite(fname, f)
        # choose encoder: prefer NVENC if available
        if ENCODER_HEVC == "hevc_nvenc":
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "frame_%06d.png"),
                "-c:v", "hevc_nvenc", "-preset", preset, "-pix_fmt", "yuv420p",
            ]
            # NVENC supports constant-quantization via -cq
            if crf is not None:
                cmd += ["-cq", str(crf)]
        else:
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "frame_%06d.png"),
                "-c:v", "libx265", "-preset", preset, "-pix_fmt", "yuv420p",
            ]
            if crf is not None:
                cmd += ["-crf", str(crf)]

        cmd.append(out_path)
        logger.debug("Encoding command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        shutil.rmtree(tmp)


def combine_frame_pairs(frames_a: list, frames_b: list, fn, out_offset: Optional[int] = 128) -> list:
    """Generic elementwise int32 operation on pairs of frames.

    - fn receives two int32 numpy arrays and must return an int32 array result.
    - If out_offset is not None it's added to the result before clipping (useful to store signed residuals).
    """
    n = min(len(frames_a), len(frames_b))
    out = []
    for i in range(n):
        a = frames_a[i].astype('int32')
        b = frames_b[i].astype('int32')
        val = fn(a, b)
        if out_offset is not None:
            val = val + out_offset
        val = val.clip(0, 255).astype('uint8')
        out.append(val)
    return out


def compute_residual_frames(gt_frames: list, up_frames: list) -> list:
    # residual = gt - up  (stored as offset uint8)
    return combine_frame_pairs(gt_frames, up_frames, lambda g, u: g - u, out_offset=128)


def apply_residuals_to_upscaled(up_frames: list, residual_frames: list):
    # reconstructed = up + (res_off - 128)
    return combine_frame_pairs(up_frames, residual_frames, lambda up, res_off: up + (res_off - 128), out_offset=None)


def sum_residual_videos(server_res_path: str, client_res_path: str, out_path: str, crf: Optional[int] = None, preset: str = "medium") -> None:
    srv_frames, srv_fps, _ = decode_video_frames(server_res_path)
    cli_frames, cli_fps, _ = decode_video_frames(client_res_path)
    fps = srv_fps or cli_fps
    # summed = (srv_off - 128) + (cli_off - 128) -> stored again with +128 offset
    summed = combine_frame_pairs(srv_frames, cli_frames, lambda a_off, b_off: (a_off - 128) + (b_off - 128), out_offset=128)
    encode_frames_to_hevc(summed, fps, out_path, crf=crf, preset=preset)


# helper: probe duration
def probe_duration(path: str) -> float:
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
def probe_height(path: str) -> Optional[int]:
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


# helper: probe fps
def probe_fps(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        if "/" in out:
            num, den = out.split("/")
            return float(num) / float(den) if float(den) != 0 else 0.0
        return float(out)
    except Exception:
        return 30.0


# helper: compute SSIM between a segment of the original and a produced chunk at a given resolution
def compute_ssim(input_path: str, chunk_path: str, start: float, dur: float, height: int) -> Optional[float]:
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


# -------------------------- Evaluation helpers --------------------------

def _filesize(path: str) -> int:
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def evaluate_and_record(gt_path: str,
                        reconstructed_path: Optional[str],
                        low_res_path: Optional[str],
                        server_residual_path: Optional[str],
                        client_residual_path: Optional[str],
                        combined_residual_path: Optional[str],
                        client_time: Optional[float],
                        server_time: Optional[float],
                        representation: int,
                        csv_path: str) -> dict:
    """Compute per-chunk evaluation and append to CSV. Returns a dict of metrics."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["chunk_index", "representation", "processing_time_client_s", "processing_time_server_s", "ssim_reconstructed_gt", "size_gt", "size_low_res", "size_server_residual", "size_client_residual", "size_combined_residual", "total_transmitted_size", "size_reconstructed"])

        # derive chunk index from filename if possible
        def _chunk_index(p: str) -> str:
            if not p:
                return ""
            m = re.search(r"chunk_(\d+)", p)
            return m.group(1) if m else ""

        chunk_idx = _chunk_index(gt_path)
        gt_size = _filesize(gt_path)
        low_size = _filesize(low_res_path)
        srv_size = _filesize(server_residual_path)
        cli_size = _filesize(client_residual_path)
        comb_size = _filesize(combined_residual_path)
        recon_size = _filesize(reconstructed_path)
        total_transmitted = low_size + comb_size

        # quality: SSIM between reconstructed and GT (if both exist)
        ssim_val = None
        if reconstructed_path and os.path.exists(reconstructed_path) and gt_path and os.path.exists(gt_path):
            dur = probe_duration(gt_path)
            h = probe_height(gt_path) or 0
            ssim_val = compute_ssim(gt_path, reconstructed_path, 0, dur, h)

        writer.writerow([chunk_idx, representation, f"{client_time:.3f}" if client_time is not None else "", f"{server_time:.3f}" if server_time is not None else "", "" if ssim_val is None else f"{ssim_val:.6f}", gt_size, low_size, srv_size, cli_size, comb_size, total_transmitted, recon_size])

    return {
        "chunk_index": chunk_idx,
        "representation": representation,
        "client_time": client_time,
        "server_time": server_time,
        "ssim": ssim_val,
        "size_gt": gt_size,
        "size_low_res": low_size,
        "size_server_residual": srv_size,
        "size_client_residual": cli_size,
        "size_combined_residual": comb_size,
        "total_transmitted": total_transmitted,
        "size_reconstructed": recon_size,
    }


# -------------------------- Client class --------------------------
class Client:
    """Simulates a simple video streaming client.

    Behavior:
      - ABR: first requested chunk is lowest representation (360p), then increase
        one level per chunk until maximum representation is reached.
      - For each received chunk it decodes frames with OpenCV, upsamples to the
        ground-truth resolution with bicubic, computes residuals (GT - upscaled),
        and encodes residual frames to a video at the ground-truth resolution.
    """

    def __init__(self, representations: Optional[List[int]] = None, gt_height: int = 2160, tmp_root: Optional[str] = None, ffmpeg_preset: str = "medium"):
        if representations is None:
            representations = [360, 720, 1080, 1440]
        self.representations = representations
        self.gt_height = gt_height
        self.ffmpeg_preset = ffmpeg_preset
        self.tmp_root = tmp_root or tempfile.mkdtemp(prefix="oracl_client_")
        logger.info("Client initialized with tmp dir %s", self.tmp_root)

    def request_representation(self, chunk_index: int) -> int:
        # Increase one level per chunk until max
        idx = min(chunk_index, len(self.representations) - 1)
        return self.representations[idx]

    def _decode(self, path: str):
        return decode_video_frames(path)

    def upscale_frames(self, frames: list, target_size: tuple) -> list:
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV and numpy are required for decoding/upscaling. Please install `opencv-python` and `numpy`.")
        target_w, target_h = target_size
        up = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_CUBIC) for f in frames]
        return up

    def compute_residuals(self, gt_frames: list, up_frames: list) -> list:
        return compute_residual_frames(gt_frames, up_frames)

    def encode_residuals(self, residual_frames: list, fps: float, out_path: str, crf: Optional[int] = None) -> None:
        encode_frames_to_hevc(residual_frames, fps, out_path, crf=crf, preset=self.ffmpeg_preset)

    def process_chunk(self, chunk_path: str, original_chunk_path: str, chunk_index: int, out_dir: str, representation: int, server: Optional['Server'] = None) -> dict:
        """Process a single chunk and return detailed results.

        Returned dict keys:
          - 'client_residual': path to client residual video
          - 'combined_residual': path to combined residual from server (or None)
          - 'server_combine_time': time in seconds spent on server combining residuals (or None)
          - 'reconstructed': path to reconstructed video on client (or None)
          - 'client_processing_time': total time this method took (s)
          - 'error': error message if failed
        """
        t0_total = time.perf_counter()
        logger.info("Client processing chunk %d: %s", chunk_index, chunk_path)
        try:
            low_frames, low_fps, (lw, lh) = self._decode(chunk_path)
            gt_frames, gt_fps, (gw, gh) = self._decode(original_chunk_path)
        except Exception as e:
            logger.error("Failed to decode chunk(s): %s", e)
            return {"error": str(e)}

        # Use GT fps and size
        target_size = (gw, gh)
        up_frames = self.upscale_frames(low_frames, target_size)
        residuals = self.compute_residuals(gt_frames, up_frames)

        os.makedirs(out_dir, exist_ok=True)
        client_residual_out = os.path.join(out_dir, f"client_residual_chunk_{chunk_index}.mp4")
        try:
            self.encode_residuals(residuals, gt_fps, client_residual_out)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to encode residuals for chunk %d: %s", chunk_index, e.stderr)
            return {"error": str(e)}

        result = {
            "client_residual": client_residual_out,
            "combined_residual": None,
            "server_combine_time": None,
            "reconstructed": None,
            "client_processing_time": None,
            "error": None,
        }

        # If no server provided, return info about client residual only
        if server is None:
            result["client_processing_time"] = time.perf_counter() - t0_total
            return result

        # Upload client residuals to server and get combined residuals back
        combined_residual_path, server_time = server.handle_client_residual(client_residual_out, representation, chunk_index)
        if combined_residual_path is None:
            logger.error("Server failed to produce combined residual for chunk %d", chunk_index)
            result["server_combine_time"] = server_time
            result["client_processing_time"] = time.perf_counter() - t0_total
            result["error"] = "server_fail"
            return result

        result["combined_residual"] = combined_residual_path
        result["server_combine_time"] = server_time

        # Client decodes combined residuals and applies them to upscaled frames
        comb_frames, comb_fps, _ = decode_video_frames(combined_residual_path)
        reconstructed = apply_residuals_to_upscaled(up_frames, comb_frames)

        # write reconstructed video for inspection
        reconstructed_out = os.path.join(out_dir, f"reconstructed_chunk_{chunk_index}.mp4")
        encode_frames_to_hevc(reconstructed, gt_fps, reconstructed_out, crf=None, preset=self.ffmpeg_preset)

        result["reconstructed"] = reconstructed_out
        result["client_processing_time"] = time.perf_counter() - t0_total
        return result

    def cleanup(self):
        try:
            shutil.rmtree(self.tmp_root)
            logger.info("Client tmp dir removed: %s", self.tmp_root)
        except Exception:
            pass


# -------------------------- Server class --------------------------
class Server:
    """Server-side pipeline: creates chunks if missing, computes server residuals, and
    accepts client residuals to produce combined residuals."""

    def __init__(self, output_dir: str = "output", resolutions: Optional[List[int]] = None, crfs: Optional[Dict[int, int]] = None, preset: str = "medium"):
        self.output_dir = output_dir
        self.resolutions = resolutions or [360, 720, 1080, 1440]
        self.crfs = crfs or {360: 28, 720: 23, 1080: 20, 1440: 18}
        self.preset = preset
        self.server_res_dir = os.path.join(self.output_dir, "server_residuals")
        os.makedirs(self.server_res_dir, exist_ok=True)

    def setup(self, input_file: str, chunk_seconds: int = 4, metrics_csv: Optional[str] = None) -> Dict[int, List[str]]:
        """Create per-chunk originals and per-resolution chunks using shared helpers.

        Also writes a CSV with per-chunk metrics and computes server residuals for each
        representation and chunk.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if metrics_csv is None:
            metrics_csv = os.path.join(self.output_dir, "metrics.csv")

        need_header = not os.path.exists(metrics_csv)
        csv_file = open(metrics_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if need_header:
            csv_writer.writerow(["resolution", "chunk_name", "file_size", "encoding_time", "ssim_score"])
            csv_file.flush()

        total_dur = probe_duration(input_file)
        if total_dur <= 0:
            raise RuntimeError("Unable to determine input duration or duration is zero")

        results: Dict[int, List[str]] = {h: [] for h in self.resolutions}
        results[0] = []  # originals

        orig_height = probe_height(input_file)
        original_dir = os.path.join(self.output_dir, "original")
        os.makedirs(original_dir, exist_ok=True)

        total_chunks = math.ceil(total_dur / chunk_seconds)
        logger.info("Server.setup: creating %d chunks of %ds each from %s", total_chunks, chunk_seconds, input_file)

        # quick existence check: if all originals and per-resolution chunks exist, skip generation
        def _all_chunks_exist() -> bool:
            # check originals
            for i in range(total_chunks):
                if not os.path.exists(os.path.join(original_dir, f"chunk_{i}.mp4")):
                    return False
            # check each representation
            for height in self.resolutions:
                for i in range(total_chunks):
                    if not os.path.exists(os.path.join(self.output_dir, f"{height}p", f"chunk_{i}.mp4")):
                        return False
            return True

        if _all_chunks_exist():
            logger.info("All chunks already exist — skipping recreation")
            # populate results from existing files
            for i in range(total_chunks):
                results[0].append(os.path.join(original_dir, f"chunk_{i}.mp4"))
            for height in self.resolutions:
                res_dir = os.path.join(self.output_dir, f"{height}p")
                os.makedirs(res_dir, exist_ok=True)
                files = [os.path.join(res_dir, f"chunk_{i}.mp4") for i in range(total_chunks) if os.path.exists(os.path.join(res_dir, f"chunk_{i}.mp4"))]
                results[height] = sorted(files)
        else:
            for chunk_idx in range(total_chunks):
                start = chunk_idx * chunk_seconds
                dur = min(chunk_seconds, max(0.0, total_dur - start))
                if dur <= 0:
                    break

                # original chunk
                out_original = os.path.join(original_dir, f"chunk_{chunk_idx}.mp4")
                if not os.path.exists(out_original):
                    copy_cmd = [
                        "ffmpeg", "-y", "-ss", str(start), "-i", input_file,
                        "-t", str(dur), "-c", "copy", out_original,
                    ]
                    try:
                        subprocess.run(copy_cmd, check=True, capture_output=True, text=True)
                        logger.info("Extracted original chunk %d (copy) -> %s", chunk_idx, out_original)
                    except subprocess.CalledProcessError:
                        # fallback to lossless re-encode
                        reenc_cmd = [
                            "ffmpeg", "-y", "-ss", str(start), "-i", input_file,
                            "-t", str(dur), "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "lossless=1",
                            "-c:a", "copy", out_original,
                        ]
                        logger.warning("Stream copy failed for original chunk %d, lossless re-encoding", chunk_idx)
                        subprocess.run(reenc_cmd, check=True, capture_output=True, text=True)

                if os.path.exists(out_original):
                    results[0].append(out_original)
                    file_size = os.path.getsize(out_original)
                    csv_writer.writerow([str(orig_height) if orig_height is not None else "original", os.path.join("original", f"chunk_{chunk_idx}.mp4"), str(file_size), "0.000", "1.000000"])
                    csv_file.flush()

                # probe fps for original chunk (avoid full decode)
                try:
                    gt_fps = probe_fps(out_original)
                except Exception:
                    gt_fps = 30.0

                for height in self.resolutions:
                    res_dir = os.path.join(self.output_dir, f"{height}p")
                    os.makedirs(res_dir, exist_ok=True)
                    out_file = os.path.join(res_dir, f"chunk_{chunk_idx}.mp4")
                    if os.path.exists(out_file):
                        results[height].append(out_file)
                        continue

                    # Use ffmpeg scaling + encoder (NVENC if available) to avoid Python per-frame work
                    vf_scale = f"scale=-2:{height}"
                    codec = ENCODER_HEVC
                    cmd = ["ffmpeg", "-y", "-i", out_original, "-vf", vf_scale, "-c:v", codec, "-preset", self.preset, "-pix_fmt", "yuv420p"]
                    rep_crf = self.crfs.get(height)
                    if codec == "hevc_nvenc":
                        if rep_crf is not None:
                            cmd += ["-rc", "vbr_hq", "-cq", str(max(0, min(51, int(rep_crf))))]
                    else:
                        if rep_crf is not None:
                            cmd += ["-crf", str(rep_crf)]

                    cmd.append(out_file)

                    t0 = time.perf_counter()
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        encoding_time = time.perf_counter() - t0
                    except subprocess.CalledProcessError as e:
                        logger.error("ffmpeg scaling/encode failed for %dp chunk %d: %s", height, chunk_idx, e.stderr)
                        continue

                    if not os.path.exists(out_file):
                        logger.error("Failed to create chunk %d at %dp", chunk_idx, height)
                        continue

                    file_size = os.path.getsize(out_file)
                    ssim = compute_ssim(input_file, out_file, start, dur, height)

                    csv_writer.writerow([str(height), os.path.join(f"{height}p", f"chunk_{chunk_idx}.mp4"), str(file_size), f"{encoding_time:.3f}", "" if ssim is None else f"{ssim:.6f}"])
                    csv_file.flush()

                    results[height].append(out_file)

        # compute server residuals now
        orig_chunks = results.get(0, [])
        num = len(orig_chunks)
        logger.info("Server: computing server residuals for %d chunks", num)
        for rep in self.resolutions:
            rep_dir = os.path.join(self.server_res_dir, f"{rep}p")
            os.makedirs(rep_dir, exist_ok=True)
            for i in range(num):
                out_path = os.path.join(rep_dir, f"server_residual_{i}.mp4")
                if os.path.exists(out_path):
                    continue
                gt_chunk = orig_chunks[i]
                low_chunk = os.path.join(self.output_dir, f"{rep}p", f"chunk_{i}.mp4")
                if not os.path.exists(low_chunk):
                    logger.warning("Missing low-res chunk for %dp/%d; skipping", rep, i)
                    continue
                logger.info("Server: computing residual for rep %dp chunk %d", rep, i)
                self.compute_server_residual(gt_chunk, low_chunk, out_path, crf=self.crfs.get(rep))

        csv_file.close()
        return results

    def compute_server_residual(self, gt_chunk_path: str, low_chunk_path: str, out_path: str, crf: Optional[int] = None) -> None:
        # decode both, upscale low to GT, compute residual frames, encode
        gt_frames, gt_fps, (gw, gh) = decode_video_frames(gt_chunk_path)
        low_frames, low_fps, (lw, lh) = decode_video_frames(low_chunk_path)
        up_frames = [cv2.resize(f, (gw, gh), interpolation=cv2.INTER_CUBIC) for f in low_frames]
        residuals = compute_residual_frames(gt_frames, up_frames)
        encode_frames_to_hevc(residuals, gt_fps, out_path, crf=crf, preset=self.preset)

    def get_server_residual_path(self, rep: int, chunk_index: int) -> Optional[str]:
        path = os.path.join(self.server_res_dir, f"{rep}p", f"server_residual_{chunk_index}.mp4")
        return path if os.path.exists(path) else None

    def handle_client_residual(self, client_residual_path: str, rep: int, chunk_index: int) -> Tuple[Optional[str], Optional[float]]:
        """Sum server residual and client residual to form combined residuals.

        Returns (combined_path or None, combine_time_seconds or None).
        """
        srv = self.get_server_residual_path(rep, chunk_index)
        if srv is None:
            logger.error("Server residual not available for rep %dp chunk %d", rep, chunk_index)
            return None, None
        out_path = os.path.join(self.server_res_dir, f"{rep}p", f"combined_residual_{chunk_index}.mp4")
        # if already exists, return it with zero time
        if os.path.exists(out_path):
            return out_path, 0.0
        # sum and encode using CRF equal to chunk CRF
        crf = self.crfs.get(rep)
        t0 = time.perf_counter()
        sum_residual_videos(srv, client_residual_path, out_path, crf=crf, preset=self.preset)
        t1 = time.perf_counter()
        return out_path, (t1 - t0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CRF-based HEVC ladder and 4s mp4 chunks using libx265")
    parser.add_argument("input", help="Input video file (e.g., input.mp4)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--chunk-seconds", type=int, default=4, help="Chunk length in seconds")
    parser.add_argument("--crf", type=int, help="Global CRF to use for all resolutions (overrides defaults)")
    parser.add_argument("--simulate-client", action="store_true", help="Run the simple client simulation on produced chunks")
    parser.add_argument("--client-output", default="client_output", help="Directory where client residuals will be written")
    args = parser.parse_args()

    crfs = None
    if args.crf is not None:
        crfs = {360: args.crf, 720: args.crf, 1080: args.crf, 1440: args.crf}

    server = Server(output_dir=args.output, crfs=crfs, preset="medium")
    res = server.setup(args.input, chunk_seconds=args.chunk_seconds)
    for h, chunks in res.items():
        print(f"{h}p: {len(chunks)} chunks -> first chunk: {chunks[0] if chunks else 'none'}")

    if args.simulate_client:
        client = Client()
        # evaluation CSV in client output
        eval_csv = os.path.join(args.client_output, "evaluation.csv")
        os.makedirs(args.client_output, exist_ok=True)

        # number of chunks determined from original list
        original_chunks = res.get(0, [])
        n = len(original_chunks)
        for i in range(n):
            requested = client.request_representation(i)
            avail = res.get(requested, [])
            if i >= len(avail):
                logger.warning("No chunk %d at %dp available, skipping", i, requested)
                continue
            chunk_path = avail[i]
            ori_path = original_chunks[i]
            out_dir = os.path.join(args.client_output, f"chunk_{i}")
            os.makedirs(out_dir, exist_ok=True)

            info = client.process_chunk(chunk_path, ori_path, i, out_dir, requested, server)
            if info.get("error"):
                logger.error("Chunk %d processing failed: %s", i, info.get("error"))
                continue

            client_res = info.get("client_residual")
            combined_res = info.get("combined_residual")
            server_res = server.get_server_residual_path(requested, i)
            reconstructed = info.get("reconstructed")
            client_time = info.get("client_processing_time")
            server_time = info.get("server_combine_time")

            metrics = evaluate_and_record(ori_path, reconstructed, chunk_path, server_res, client_res, combined_res, client_time, server_time, requested, eval_csv)
            logger.info("Chunk %d eval: SSIM=%s total_sent=%d bytes", i, metrics.get("ssim"), metrics.get("total_transmitted"))
        client.cleanup()

