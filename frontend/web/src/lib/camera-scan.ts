type Point = { x: number; y: number };

export const FACE_OUTLINE = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

type Features = {
  lengthToWidth: number; foreheadToCheek: number; jawToCheek: number;
  foreheadToJaw: number; chinToJaw: number; jawAngle: number;
};

const PROTOTYPES = {
  oval: { lengthToWidth: [1.22, .06], foreheadToJaw: [1.10, .08], jawToCheek: [.78, .05], jawAngle: [148, 7] },
  round: { lengthToWidth: [1.08, .06], foreheadToJaw: [1.05, .08], jawToCheek: [.82, .05], jawAngle: [152, 7] },
  square: { lengthToWidth: [1.10, .06], foreheadToJaw: [.98, .07], jawToCheek: [.88, .05], jawAngle: [141, 7] },
  oblong: { lengthToWidth: [1.36, .07], foreheadToJaw: [1.05, .08], jawToCheek: [.82, .06], jawAngle: [147, 8] },
  heart: { lengthToWidth: [1.20, .07], foreheadToJaw: [1.25, .08], jawToCheek: [.72, .05], chinToJaw: [.38, .06] },
  diamond: { lengthToWidth: [1.25, .07], foreheadToCheek: [.80, .05], foreheadToJaw: [1, .08], jawToCheek: [.75, .05] },
};

const clampUnit = (value: number) => Math.max(-1, Math.min(1, value));
const degrees = (radians: number) => radians * 180 / Math.PI;

function distance(a: Point, b: Point, width: number, height: number) {
  return Math.hypot((a.x - b.x) * width, (a.y - b.y) * height);
}

export function estimateFaceShape(points: Point[], width: number, height: number): string | null {
  const dist = (a: number, b: number) => distance(points[a], points[b], width, height);
  const angle = (a: number, vertex: number, b: number) => {
    const sideA = dist(a, vertex), sideB = dist(b, vertex), opposite = dist(a, b);
    if (!sideA || !sideB) return NaN;
    return degrees(Math.acos(clampUnit((sideA ** 2 + sideB ** 2 - opposite ** 2) / (2 * sideA * sideB))));
  };
  const cheek = dist(234, 454), jaw = dist(172, 397), forehead = dist(54, 284);
  const features: Features = {
    lengthToWidth: dist(10, 152) / cheek,
    foreheadToCheek: forehead / cheek,
    jawToCheek: jaw / cheek,
    foreheadToJaw: forehead / jaw,
    chinToJaw: dist(176, 400) / jaw,
    jawAngle: (angle(132, 172, 148) + angle(361, 397, 377)) / 2,
  };
  if (!Object.values(features).every(Number.isFinite)) return null;
  let best: string | null = null;
  let min = Infinity;
  for (const [shape, prototype] of Object.entries(PROTOTYPES)) {
    const terms = Object.entries(prototype) as [keyof Features, number[]][];
    const score = terms.reduce((sum, [key, [mean, tolerance]]) =>
      sum + ((features[key] - mean) / tolerance) ** 2, 0) / terms.length;
    if (score < min) { best = shape; min = score; }
  }
  return best;
}

export function faceBounds(points: Point[]) {
  const outline = FACE_OUTLINE.map((index) => points[index]);
  return {
    left: Math.min(...outline.map((p) => p.x)), right: Math.max(...outline.map((p) => p.x)),
    top: Math.min(...outline.map((p) => p.y)), bottom: Math.max(...outline.map((p) => p.y)),
  };
}

export function measureLightAndSharpness({ data, width, height }: ImageData) {
  const pixels = new Float32Array(width * height);
  let total = 0;
  for (let i = 0; i < pixels.length; i++) {
    const j = i * 4;
    pixels[i] = .299 * data[j] + .587 * data[j + 1] + .114 * data[j + 2];
    total += pixels[i];
  }
  const brightness = total / pixels.length;
  let sum = 0, squares = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const laplacian = pixels[i - 1] + pixels[i + 1] + pixels[i - width] + pixels[i + width] - 4 * pixels[i];
      sum += laplacian;
      squares += laplacian ** 2;
    }
  }
  const count = (width - 2) * (height - 2);
  const sharpness = count > 0 ? (squares / count - (sum / count) ** 2) * (128 / Math.max(brightness, 1)) ** 2 : 0;
  return { brightness, sharpness };
}

export function assessCameraFrame(
  points: Point[], width: number, height: number, matrix: number[],
  { brightness, sharpness }: ReturnType<typeof measureLightAndSharpness>,
): { hint: string; score: number | null; frontal: boolean } {
  const { left, right, top, bottom } = faceBounds(points);
  const centerOffset = Math.max(Math.abs((left + right) / 2 - .5), Math.abs((top + bottom) / 2 - .5));
  const faceWidth = distance(points[234], points[454], width, height);
  const pitch = degrees(Math.atan2(matrix[6], matrix[10]));
  const yaw = degrees(Math.atan2(-matrix[2], Math.hypot(matrix[6], matrix[10])));
  const roll = degrees(Math.atan2(matrix[1], matrix[0]));
  const pose = Math.max(Math.abs(yaw) / 15, Math.abs(pitch) / 25, Math.abs(roll) / 25);
  const frontal = pose <= 1;
  const unavailable = (hint: string) => ({ hint, score: null, frontal });
  if (Math.min(width, height) < 256) return unavailable("Resolusi kamera terlalu rendah. Coba unggah foto.");
  if (left < .02 || right > .98 || top < .02 || bottom > .98) return unavailable("Mundur sedikit agar seluruh wajah terlihat.");
  if (centerOffset > .18) return unavailable("Posisikan wajah di tengah kamera.");
  if (faceWidth < 150) return unavailable("Dekatkan wajah ke kamera.");
  if (!frontal) return unavailable("Hadapkan wajah lurus dan sejajarkan kamera dengan mata.");
  if (brightness < 50) return unavailable("Cari tempat dengan pencahayaan lebih terang.");
  if (brightness > 215) return unavailable("Kurangi cahaya langsung ke wajah.");
  if (sharpness < 15) return unavailable("Wajah tampak buram. Tahan posisi sebentar.");
  const score = 1 - centerOffset + (1 - pose) + (1 - Math.abs(brightness - 128) / 128) + Math.min(sharpness / 100, 1);
  return { hint: "Posisi wajah dan cahaya sudah sesuai.", score, frontal };
}
