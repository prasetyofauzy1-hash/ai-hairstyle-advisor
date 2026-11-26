import type { FaceLandmarker } from "@mediapipe/tasks-vision";

type Detector = Pick<FaceLandmarker, "detectForVideo" | "close">;
type Delegate = "GPU" | "CPU";

export async function createCameraDetector(create: (delegate: Delegate) => Promise<Detector>) {
  let delegate: Delegate = "GPU";
  let detector: Detector | undefined;
  let closed = false;

  function dispose() {
    const current = detector;
    detector = undefined;
    try {
      current?.close();
    } catch {
      // A failed WebGL context can also throw during teardown.
    }
  }

  async function startCpu() {
    delegate = "CPU";
    const cpu = await create(delegate);
    detector = cpu;
    if (closed) dispose();
  }

  try {
    detector = await create(delegate);
  } catch {
    await startCpu();
  }

  return {
    async detectForVideo(frame: HTMLCanvasElement, timestamp: number) {
      try {
        return detector?.detectForVideo(frame, timestamp);
      } catch (error) {
        if (delegate === "CPU") throw error;
        dispose();
        await startCpu();
        return detector?.detectForVideo(frame, timestamp);
      }
    },
    close() {
      closed = true;
      dispose();
    },
  };
}
