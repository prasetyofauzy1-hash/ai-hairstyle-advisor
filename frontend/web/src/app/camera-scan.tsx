"use client";

import { useEffect, useRef, useState } from "react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { SHAPES } from "@/lib/api";
import { assessCameraFrame, estimateFaceShape, faceBounds, FACE_OUTLINE, measureLightAndSharpness } from "@/lib/camera-scan";

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const FALLBACK = "Gunakan pilihan unggah foto.";
const CHOOSE = "Tahan posisi sebentar untuk memilih foto terbaik.";
const SAMPLE_MS = 100;
const SELECTION_MS = 1500;

type Scan = { hint: string; shape: string | null; ready: boolean };
const INITIAL: Scan = { hint: "Menyalakan kamera…", shape: null, ready: false };

function cameraError(error: unknown) {
  const name = (error as Error | undefined)?.name;
  if (name === "NotAllowedError" || name === "SecurityError") return "Izin kamera ditolak.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "Kamera tidak ditemukan.";
  return "Kamera tidak bisa dibuka. Periksa apakah kamera sedang digunakan aplikasi lain.";
}

export default function CameraScan({ onCapture, onCancel }: { onCapture: (photo: File) => void; onCancel: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const outlineRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<(() => void) | null>(null);
  const onCaptureRef = useRef(onCapture);
  const [scan, setScan] = useState(INITIAL);
  const [error, setError] = useState("");
  const [capturing, setCapturing] = useState(false);

  useEffect(() => { onCaptureRef.current = onCapture; }, [onCapture]);

  useEffect(() => {
    const video = videoRef.current!;
    const outline = outlineRef.current!;
    const overlay = outline.getContext("2d");
    const frame = document.createElement("canvas");
    let best = document.createElement("canvas");
    let selected = document.createElement("canvas");
    const sample = document.createElement("canvas");
    const frameContext = frame.getContext("2d");
    let bestContext = best.getContext("2d");
    let selectedContext = selected.getContext("2d");
    const sampleContext = sample.getContext("2d", { willReadFrequently: true });
    let active = true;
    let stream: MediaStream | undefined;
    let landmarker: FaceLandmarker | undefined;
    let animation = 0;
    let previousVideoTime = -1;
    let nextSample = 0;
    let startedAt: number | null = null;
    let bestScore = -Infinity;
    let ready = false;

    function resetSelection() {
      startedAt = null;
      bestScore = -Infinity;
      ready = false;
    }

    function stop() {
      active = false;
      cancelAnimationFrame(animation);
      captureRef.current = null;
      stream?.getTracks().forEach((track) => track.stop());
      stream = undefined;
      video.srcObject = null;
      landmarker?.close();
      landmarker = undefined;
    }

    function fail(message: string) {
      if (!active) return;
      stop();
      setError(`${message} ${FALLBACK}`);
    }

    function tick(now: number) {
      if (!active || !landmarker || !frameContext || !bestContext || !sampleContext || !overlay) return;
      animation = requestAnimationFrame(tick);
      if (now < nextSample || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.currentTime === previousVideoTime) return;
      previousVideoTime = video.currentTime;
      nextSample = now + SAMPLE_MS;
      const width = video.videoWidth, height = video.videoHeight;
      if (!width || !height) return;
      if (frame.width !== width || frame.height !== height) {
        frame.width = outline.width = width;
        frame.height = outline.height = height;
        resetSelection();
      }
      frameContext.drawImage(video, 0, 0, width, height);
      try {
        const { faceLandmarks, facialTransformationMatrixes } = landmarker.detectForVideo(frame, now);
        overlay.clearRect(0, 0, width, height);
        const points = faceLandmarks[0];
        const matrix = facialTransformationMatrixes[0]?.data;
        let hint = "Wajah belum terdeteksi. Lihat ke arah kamera.";
        let shape: string | null = null;
        if (faceLandmarks.length === 1 && matrix?.length === 16) {
          const { left, right, top, bottom } = faceBounds(points);
          const x = Math.max(0, left), y = Math.max(0, top);
          const cropWidth = (Math.min(1, right) - x) * width;
          const cropHeight = (Math.min(1, bottom) - y) * height;
          if (cropWidth > 0 && cropHeight > 0) {
            sample.width = 256;
            sample.height = Math.max(3, Math.min(512, Math.round(256 * cropHeight / cropWidth)));
            sampleContext.drawImage(frame, x * width, y * height, cropWidth, cropHeight, 0, 0, sample.width, sample.height);
            const metrics = measureLightAndSharpness(sampleContext.getImageData(0, 0, sample.width, sample.height));
            const quality = assessCameraFrame(points, width, height, matrix, metrics);
            hint = quality.hint;
            if (quality.frontal) shape = estimateFaceShape(points, width, height);
            overlay.beginPath();
            FACE_OUTLINE.forEach((index, i) => {
              const { x: px, y: py } = points[index];
              if (i === 0) overlay.moveTo(px * width, py * height);
              else overlay.lineTo(px * width, py * height);
            });
            overlay.closePath();
            overlay.lineWidth = Math.max(3, width / 230);
            overlay.strokeStyle = quality.score === null ? "#fbbf24" : "#34d399";
            overlay.stroke();

            if (quality.score !== null) {
              startedAt ??= now;
              if (quality.score > bestScore) {
                if (best.width !== width || best.height !== height) {
                  best.width = width;
                  best.height = height;
                }
                bestContext.drawImage(frame, 0, 0);
                bestScore = quality.score;
              }
              if (now - startedAt >= SELECTION_MS) {
                [best, selected] = [selected, best];
                [bestContext, selectedContext] = [selectedContext, bestContext];
                startedAt = null;
                bestScore = -Infinity;
                ready = true;
              }
              if (!ready) hint = CHOOSE;
            } else {
              resetSelection();
            }
          } else {
            resetSelection();
          }
        } else {
          resetSelection();
          if (faceLandmarks.length > 1) hint = "Pastikan hanya satu wajah terlihat di kamera.";
        }
        setScan((last) => last.hint === hint && last.shape === shape && last.ready === ready ? last : { hint, shape, ready });
      } catch {
        fail("Pemindaian kamera terhenti.");
      }
    }

    async function start() {
      if (!navigator.mediaDevices?.getUserMedia || !overlay || !frameContext || !bestContext || !selectedContext || !sampleContext) {
        fail("Kamera tidak tersedia di peramban ini.");
        return;
      }
      try {
        const media = await navigator.mediaDevices.getUserMedia({
          audio: false, video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        if (!active) {
          media.getTracks().forEach((track) => track.stop());
          return;
        }
        stream = media;
        stream.getVideoTracks().forEach((track) => track.addEventListener("ended", () => fail("Kamera terputus."), { once: true }));
        video.srcObject = stream;
        await video.play();
      } catch (error) {
        fail(cameraError(error));
        return;
      }
      if (!active) return;
      setScan({ ...INITIAL, hint: "Memuat pemindai wajah…" });
      try {
        const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
        const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
        if (!active) return;
        const detector = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
          runningMode: "VIDEO",
          numFaces: 2,
          outputFacialTransformationMatrixes: true,
        });
        if (!active) {
          detector.close();
          return;
        }
        landmarker = detector;
        animation = requestAnimationFrame(tick);
      } catch {
        fail("Pemindai wajah tidak bisa dimuat. Periksa koneksi internet.");
      }
    }

    captureRef.current = () => {
      if (!active || !ready) return;
      captureRef.current = null;
      cancelAnimationFrame(animation);
      setCapturing(true);
      selected.toBlob((blob) => {
        if (!active) return;
        if (!blob) return fail("Foto kamera tidak bisa dibuat.");
        stop();
        onCaptureRef.current(new File([blob], "hasil-pindai.jpg", { type: "image/jpeg" }));
      }, "image/jpeg", .9);
    };
    const leave = () => fail("Kamera telah dihentikan.");
    window.addEventListener("pagehide", leave);
    void start();
    return () => {
      window.removeEventListener("pagehide", leave);
      stop();
    };
  }, []);

  return (
    <div className="mt-4 space-y-3">
      <p className="text-xs text-[var(--muted)]">Pemindaian berlangsung di perangkat ini. Hanya foto yang dipilih yang dikirim saat Anda menekan Analisis wajah.</p>
      <div className={`relative overflow-hidden rounded-lg bg-stone-950 ${error ? "hidden" : ""}`}>
        <video ref={videoRef} autoPlay muted playsInline aria-label="Pratinjau kamera" className="min-h-40 w-full -scale-x-100 object-fill" />
        <canvas ref={outlineRef} aria-hidden="true" className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" />
      </div>
      {error ? (
        <p role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">{error}</p>
      ) : (
        <>
          <p aria-live="polite" className="text-sm text-[var(--muted)]">{scan.hint}</p>
          <p className="text-sm font-medium">Perkiraan bentuk saat memindai: {scan.shape ? SHAPES[scan.shape] : "Belum tersedia"}</p>
        </>
      )}
      <div className="flex flex-wrap gap-2">
        {!error && (
          <button type="button" disabled={!scan.ready || capturing} onClick={() => captureRef.current?.()}
            className="min-h-11 rounded-lg bg-[var(--accent)] px-4 py-3 text-sm font-medium text-[var(--accent-fg)] disabled:opacity-40">
            {capturing ? "Menyiapkan foto…" : "Gunakan foto terbaik"}
          </button>
        )}
        <button type="button" onClick={onCancel} className="min-h-11 rounded-lg border border-[var(--border)] px-4 py-3 text-sm font-medium">
          Kembali ke unggah foto
        </button>
      </div>
    </div>
  );
}
