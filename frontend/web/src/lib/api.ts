export type QualityIssue = { code: string; message: string; blocking: boolean };
export type Prediction = { shape: string; label_id: string; scores: Record<string, number>; method: string; calibrated: boolean };
export type Analysis = {
  id: string;
  access_token?: string;
  image_stored: boolean;
  consent_given_at: string | null;
  quality: { passed: boolean; issues: QualityIssue[]; metrics: Record<string, number> };
  features: Record<string, unknown> | null;
  forehead_visible: boolean | null;
  prediction: Prediction | null;
  rule_baseline: { shape: string; label_id?: string } | null;
  limitations: string[];
};
export type ScoreTerm = { rule: string; points: number; reason: string };
export type Rec = {
  rank: number; id: string; name: string; description: string; score: number; score_unit: string;
  score_breakdown: ScoreTerm[]; explanation: string; advantages: string[]; styling_tips: string[];
  length: string; maintenance: string; reference_image: unknown; reference_image_note: string | null;
};
export type Recs = { face_shape: string; face_shape_label: string; face_shape_source: "model" | "user"; recommendations: Rec[]; notes: string[] };
export type RecRequest = {
  hair_type: string; thickness: string; gender?: string | null; preferred_length?: string | null;
  max_maintenance?: string | null; avoid_bangs?: boolean; formal?: boolean; face_shape?: string | null;
};

async function send(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch {
    throw new Error("Tidak bisa terhubung ke server. Periksa koneksi internet lalu coba lagi.");
  }
}

async function check<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : "Terjadi kesalahan. Coba lagi.";
    throw new Error(detail);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}

export async function createAnalysis(file: File, consent: boolean): Promise<Analysis> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("consent", String(consent));
  return check(await send("/api/analyses", { method: "POST", body: fd }));
}

export async function getRecommendations(id: string, token: string, req: RecRequest): Promise<Recs> {
  return check(await send(`/api/analyses/${id}/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Analysis-Token": token },
    body: JSON.stringify(req),
  }));
}

export async function deleteAnalysis(id: string, token: string): Promise<void> {
  return check(await send(`/api/analyses/${id}`, { method: "DELETE", headers: { "X-Analysis-Token": token } }));
}

export const SHAPES: Record<string, string> = {
  oval: "Oval", round: "Bulat", square: "Persegi", oblong: "Lonjong", heart: "Hati", diamond: "Wajik (Diamond)",
};
export const SHAPE_HINTS: Record<string, string> = {
  oval: "Oval (sedikit lebih panjang dari lebar, dagu membulat)",
  round: "Bulat (panjang dan lebar hampir sama, pipi penuh)",
  square: "Persegi (rahang lebar dan bersudut)",
  oblong: "Lonjong (lebih panjang dan relatif sempit)",
  heart: "Hati (dahi lebar, dagu runcing)",
  diamond: "Wajik (tulang pipi paling lebar, dahi dan dagu sempit)",
};
export const HAIR_TYPES: Record<string, string> = { straight: "Lurus", wavy: "Bergelombang", curly: "Keriting", coily: "Keriting rapat" };
export const THICKNESS: Record<string, string> = { thin: "Tipis", medium: "Sedang", thick: "Tebal" };
export const LENGTHS: Record<string, string> = { very_short: "Sangat pendek", short: "Pendek", medium: "Sedang", long: "Panjang" };
export const MAINT: Record<string, string> = { low: "Ringan", medium: "Sedang", high: "Intensif" };
