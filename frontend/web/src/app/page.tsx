"use client";

import { useEffect, useRef, useState } from "react";
import {
  Analysis, HAIR_TYPES, LENGTHS, MAINT, RecRequest, Recs, SHAPE_HINTS, SHAPES, THICKNESS,
  createAnalysis, deleteAnalysis, getRecommendations,
} from "@/lib/api";

type Saved = { id: string; token: string; createdAt: string; imageStored: boolean };
const SAVED_KEY = "hairstyle-advisor:analyses";
const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
// Mirrors the backend MAX_UPLOAD_BYTES; the server still enforces its own limit.
const MAX_BYTES = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_BYTES) || 8 * 1024 * 1024;
const UNREADABLE = "Kami tidak bisa membuka foto ini. Coba pilih foto lain berformat JPEG, PNG, atau WebP.";

function readSaved(): Saved[] {
  try { return JSON.parse(localStorage.getItem(SAVED_KEY) || "[]"); } catch { return []; }
}

const card = "rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 sm:p-6 shadow-sm";
const btn = "min-h-11 rounded-lg bg-[var(--accent)] px-4 py-3 text-sm font-medium text-[var(--accent-fg)] disabled:opacity-40";
const btnGhost = "min-h-11 rounded-lg border border-[var(--border)] px-4 py-3 text-sm font-medium";
const label = "block text-sm font-medium mb-1";
const select = "min-h-11 w-full rounded-lg border border-[var(--border)] bg-white px-3 py-2 text-sm";

function Select({ name, value, onChange, options, empty }: {
  name: string; value: string; onChange: (v: string) => void; options: Record<string, string>; empty?: string;
}) {
  return (
    <label className="block">
      <span className={label}>{name}</span>
      <select className={select} value={value} onChange={(e) => onChange(e.target.value)}>
        {empty !== undefined && <option value="">{empty}</option>}
        {Object.entries(options).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
      </select>
    </label>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [action, setAction] = useState<"analyze" | "recs" | "delete" | null>(null);
  const [storageBlocked, setStorageBlocked] = useState(false);
  const [askedToStore, setAskedToStore] = useState(false);
  const [deletedPhoto, setDeletedPhoto] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [recs, setRecs] = useState<Recs | null>(null);
  const [deleted, setDeleted] = useState(false);
  const [prefs, setPrefs] = useState<RecRequest>({
    hair_type: "straight", thickness: "medium", gender: "", preferred_length: "", max_maintenance: "",
    avoid_bangs: false, formal: false, face_shape: "",
  });

  const [saved, setSaved] = useState<Saved[]>([]);
  const [errorAt, setErrorAt] = useState<"upload" | "recs">("upload");
  const [status, setStatus] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const analysisRef = useRef<HTMLHeadingElement>(null);
  const recsRef = useRef<HTMLHeadingElement>(null);
  const focusAfterRender = (ref: React.RefObject<HTMLElement>) =>
    requestAnimationFrame(() => { ref.current?.focus(); ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }); });

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    setSaved(readSaved());
    const sync = (e: StorageEvent) => { if (e.key === SAVED_KEY) setSaved(readSaved()); };
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);

  // Re-read before writing so another tab's entries are kept; fall back to memory when site storage is blocked.
  function updateSaved(update: (cur: Saved[]) => Saved[]) {
    try {
      const next = update(readSaved());
      localStorage.setItem(SAVED_KEY, JSON.stringify(next));
      setSaved(next); setStorageBlocked(false);
    } catch {
      setSaved((cur) => update(cur)); setStorageBlocked(true);
    }
  }

  function pick(f: File | null) {
    setAnalysis(null); setToken(null); setRecs(null); setError(null); setErrorAt("upload"); setDeleted(false); setConsent(false);
    setPrefs((p) => ({ ...p, face_shape: "" }));
    if (f && f.type && !ACCEPTED.includes(f.type)) return reject("Format foto harus JPEG, PNG, atau WebP.");
    if (f && f.size > MAX_BYTES) return reject(`Ukuran foto maksimal ${Math.floor(MAX_BYTES / (1024 * 1024))} MB.`);
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  function reject(msg: string) {
    setFile(null); setPreview(null); setError(msg); setErrorAt("upload");
  }

  async function analyze() {
    if (!file) return;
    setBusy(true); setAction("analyze"); setError(null); setErrorAt("upload"); setRecs(null); setStatus("Menganalisis foto...");
    setAskedToStore(consent);
    try {
      const a = await createAnalysis(file, consent);
      setStatus(a.quality.passed ? "Analisis selesai." : "Analisis selesai, tetapi foto belum memenuhi syarat kualitas.");
      focusAfterRender(analysisRef);
      setAnalysis(a); setToken(a.access_token ?? null);
      if (a.access_token) {
        const entry = { id: a.id, token: a.access_token, createdAt: new Date().toISOString(), imageStored: a.image_stored };
        updateSaved((cur) => [entry, ...cur.filter((x) => x.id !== entry.id)]);
      }
    } catch (e) { setError((e as Error).message); setStatus(""); } finally { setBusy(false); }
  }

  async function recommend() {
    if (!analysis || !token) return;
    setBusy(true); setAction("recs"); setError(null); setErrorAt("recs");
    const clean = Object.fromEntries(Object.entries(prefs).map(([k, v]) => [k, v === "" ? null : v])) as RecRequest;
    try {
      const r = await getRecommendations(analysis.id, token, clean);
      setRecs(r); setStatus(`${r.recommendations.length} rekomendasi siap.`); focusAfterRender(recsRef);
    }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function remove(id: string, tok: string, hadPhoto: boolean) {
    setBusy(true); setAction("delete");
    try {
      await deleteAnalysis(id, tok);
      updateSaved((cur) => cur.filter((x) => x.id !== id));
      if (analysis?.id === id) pick(null);
      const msg = hadPhoto ? "Analisis, foto, dan rekomendasi Anda sudah dihapus dari server." : "Analisis dan rekomendasi Anda sudah dihapus dari server.";
      setDeleted(true); setDeletedPhoto(hadPhoto); setStatus(msg);
      requestAnimationFrame(() => { fileRef.current?.focus(); fileRef.current?.closest("section")?.scrollIntoView({ behavior: "smooth", block: "start" }); });
    } catch (e) { setError((e as Error).message); setStatus(""); } finally { setBusy(false); }
  }

  const set = (k: keyof RecRequest) => (v: string | boolean) => { setPrefs((p) => ({ ...p, [k]: v })); setRecs(null); };
  const pred = analysis?.prediction;
  const canRecommend = !!pred || !!prefs.face_shape;

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 sm:py-12">
      <header className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-widest text-[var(--accent)]">Proyek riset · nonkomersial</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight sm:text-4xl">AI Hairstyle Advisor</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Unggah foto wajah tampak depan. Sistem mengukur proporsi wajah, memperkirakan bentuk wajah,
          lalu merekomendasikan gaya rambut beserta alasannya.
        </p>
      </header>

      <p role="status" aria-live="polite" className="sr-only">{status}</p>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <section className={card}>
          <h2 className="text-lg font-semibold">1. Unggah foto</h2>
          {deleted && (
            <div className="mt-4 rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-900">
              {deletedPhoto ? "Analisis, foto, dan rekomendasi Anda sudah dihapus dari server." : "Analisis dan rekomendasi Anda sudah dihapus dari server."}
            </div>
          )}
          <label className="mt-4 flex aspect-[4/5] cursor-pointer focus-within:ring-2 focus-within:ring-[var(--accent)] focus-within:ring-offset-2 items-center justify-center overflow-hidden rounded-lg border-2 border-dashed border-[var(--border)] bg-stone-50">
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="Pratinjau foto" className="h-full w-full object-cover" onError={() => reject(UNREADABLE)} />
            ) : file ? (
              <span className="px-6 text-center text-sm text-[var(--muted)]">Pratinjau tidak tersedia untuk {file.name}</span>
            ) : (
              <span className="px-6 text-center text-sm text-[var(--muted)]">Klik untuk memilih foto (JPEG, PNG, atau WebP, maks. 8 MB)</span>
            )}
            <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" aria-label="Pilih foto"
              onChange={(e) => { pick(e.target.files?.[0] ?? null); e.target.value = ""; }} />
          </label>
          <p className="mt-3 text-xs text-[var(--muted)]">Tips: wajah menghadap kamera, cahaya merata, tanpa kacamata hitam, dahi terlihat.</p>

          <label className="mt-4 flex items-start gap-3 rounded-lg bg-stone-50 p-3 text-sm">
            <input type="checkbox" className="mt-0.5 h-4 w-4" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            <span>
              Saya setuju aplikasi <strong>menyimpan foto ini</strong>. Jika tidak dicentang, foto tetap dikirim untuk dianalisis,
              tetapi tidak disimpan. Hasil analisis tetap disimpan meski fotonya tidak. Foto yang belum memenuhi syarat
              kualitas tidak disimpan. Anda bisa menghapus semuanya kapan saja.
            </span>
          </label>

          <button className={`${btn} mt-4 w-full`} disabled={!file || busy} onClick={analyze}>
            {busy && action === "analyze" ? "Menganalisis..." : "Analisis wajah"}
          </button>
          {error && errorAt === "upload" && (
            <div role="alert" className="mt-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">{error}</div>
          )}
        </section>

        <section className="space-y-6">
          {!analysis && (
            <div className={`${card} text-sm text-[var(--muted)]`}>Hasil analisis akan muncul di sini.</div>
          )}

          {analysis && (
            <div className={card}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <h2 ref={analysisRef} tabIndex={-1} className="scroll-mt-4 text-lg font-semibold outline-none">2. Hasil analisis</h2>
                <button className={`${btnGhost} text-red-700`} onClick={() => token && remove(analysis.id, token, analysis.image_stored)} disabled={busy}>
                  {busy && action === "delete" ? "Menghapus..." : analysis.image_stored ? "Hapus analisis & foto" : "Hapus analisis"}
                </button>
              </div>
              <p className="mt-1 text-xs text-[var(--muted)]">
                {analysis.image_stored ? "Foto disimpan atas persetujuan Anda."
                  : askedToStore && !analysis.quality.passed ? "Foto tidak disimpan karena belum memenuhi syarat kualitas."
                  : "Foto tidak disimpan."}
              </p>

              {analysis.quality.issues.length > 0 && (
                <ul className="mt-4 space-y-2">
                  {analysis.quality.issues.map((i) => (
                    <li key={i.code} className={`rounded-lg px-3 py-2 text-sm ${i.blocking ? "bg-red-50 text-red-900" : "bg-amber-50 text-amber-900"}`}>
                      {i.message}
                    </li>
                  ))}
                </ul>
              )}

              {!analysis.quality.passed && (
                <p className="mt-4 text-sm font-medium">Foto belum memenuhi syarat kualitas, jadi aplikasi belum bisa memperkirakan bentuk wajah. Coba foto lain, atau pilih bentuk wajah sendiri di bawah.</p>
              )}

              {pred && (
                <div className="mt-4">
                  <p className="text-sm text-[var(--muted)]">Perkiraan bentuk wajah (hasil utama)</p>
                  <p className="text-3xl font-semibold">{SHAPES[pred.shape] ?? pred.label_id}</p>
                  <p className="mt-3 text-xs text-[var(--muted)]">Skor ini membandingkan bentuk wajah. Skor 0,60 bukan berarti hasilnya 60% mungkin benar.</p>
                  <div className="mt-2 space-y-1.5">
                    {Object.entries(pred.scores).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                      <div key={k} className="flex items-center gap-3 text-sm">
                        <span className="w-20 shrink-0">{SHAPES[k] ?? k}</span>
                        <div className="h-2 flex-1 rounded bg-stone-100">
                          <div className="h-2 rounded bg-[var(--accent)]" style={{ width: `${v * 100}%` }} />
                        </div>
                        <span className="w-12 text-right tabular-nums text-[var(--muted)]">{v.toFixed(2).replace(".", ",")}</span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-[var(--muted)]">
                    Skor <strong>tidak terkalibrasi</strong>: bukan peluang hasil ini benar. Detail teknis: model {pred.method}.
                  </p>
                  {analysis.rule_baseline && (
                    <p className="mt-2 text-xs text-[var(--muted)]">
                      Cara lain untuk memperkirakan bentuk wajah (aturan ukuran wajah) memberi hasil {SHAPES[analysis.rule_baseline.shape] ?? analysis.rule_baseline.shape}.
                      Keduanya hanya perkiraan. Saran gaya rambut mengikuti hasil utama kecuali Anda memilih bentuk sendiri.
                    </p>
                  )}

                </div>
              )}

              <details className="mt-4 rounded-lg bg-amber-50 p-3 text-sm text-amber-950" open>
                <summary className="flex min-h-11 cursor-pointer items-center font-medium">Keterbatasan model</summary>
                <ul className="mt-2 list-disc space-y-1 pl-5">{analysis.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
              </details>
            </div>
          )}

          {analysis && (
            <div className={card}>
              <h2 className="text-lg font-semibold">3. Profil rambut</h2>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <Select name="Jenis rambut" value={prefs.hair_type} onChange={set("hair_type")} options={HAIR_TYPES} />
                <Select name="Ketebalan rambut" value={prefs.thickness} onChange={set("thickness")} options={THICKNESS} />
                <Select name="Gaya rambut" value={prefs.gender ?? ""} onChange={set("gender")} options={{ pria: "Gaya pria", wanita: "Gaya wanita" }} empty="Semua gaya" />
                <Select name="Panjang yang diinginkan" value={prefs.preferred_length ?? ""} onChange={set("preferred_length")} options={LENGTHS} empty="Bebas" />
                <Select name="Tingkat perawatan yang sanggup Anda lakukan" value={prefs.max_maintenance ?? ""} onChange={set("max_maintenance")} options={MAINT} empty="Bebas" />
                <Select name="Bentuk wajah" value={prefs.face_shape ?? ""} onChange={set("face_shape")} options={SHAPE_HINTS}
                  empty={pred ? `Ikuti perkiraan aplikasi (${SHAPES[pred.shape]})` : "Pilih bentuk wajah"} />
              </div>
              <div className="mt-4 flex flex-wrap gap-3 text-sm">
                <label className="flex min-h-11 cursor-pointer items-center gap-3 rounded-lg border border-[var(--border)] px-3"><input type="checkbox" className="h-5 w-5" checked={!!prefs.avoid_bangs} onChange={(e) => set("avoid_bangs")(e.target.checked)} /> Hindari poni</label>
                <label className="flex min-h-11 cursor-pointer items-center gap-3 rounded-lg border border-[var(--border)] px-3"><input type="checkbox" className="h-5 w-5" checked={!!prefs.formal} onChange={(e) => set("formal")(e.target.checked)} /> Utamakan gaya untuk kantor atau acara formal</label>
              </div>
              <button className={`${btn} mt-5`} onClick={recommend} disabled={busy || !canRecommend}>
                {busy && action === "recs" ? "Menyiapkan rekomendasi..." : "Tampilkan rekomendasi"}
              </button>
              {error && errorAt === "recs" && (
                <div role="alert" className="mt-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">{error}</div>
              )}
            </div>
          )}

          {recs && (
            <div className={card}>
              <h2 ref={recsRef} tabIndex={-1} className="scroll-mt-4 text-lg font-semibold outline-none">4. Rekomendasi gaya rambut untuk wajah {recs.face_shape_label}</h2>
              <p className="mt-1 text-xs text-[var(--muted)]">
                Bentuk wajah: {recs.face_shape_source === "model" ? "perkiraan aplikasi" : "pilihan Anda"}.
              </p>
              {recs.notes.length > 0 && (
                <ul className="mt-3 space-y-1 text-sm text-amber-900">{recs.notes.map((n) => <li key={n}>• {n}</li>)}</ul>
              )}
              {recs.recommendations.length === 0 && <p className="mt-4 text-sm">Belum ada gaya yang sesuai. Coba ubah pilihan, misalnya pilih Semua gaya atau bolehkan poni.</p>}
              <ol className="mt-4 space-y-4">
                {recs.recommendations.map((r) => (
                  <li key={r.id} className="rounded-lg border border-[var(--border)] p-4">
                    <div className="flex items-baseline justify-between gap-3">
                      <h3 className="font-semibold">{r.rank}. {r.name}</h3>
                      <span className="shrink-0 rounded bg-stone-100 px-2 py-0.5 text-xs tabular-nums">{r.score} {r.score_unit}</span>
                    </div>
                    <p className="mt-1 text-xs text-[var(--muted)]">Panjang: {r.length} · Perawatan: {r.maintenance}</p>
                    <p className="mt-2 text-sm">{r.description}</p>
                    <p className="mt-2 text-sm"><strong>Alasan dan hal yang perlu diperhatikan:</strong> {r.explanation}</p>
                    <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
                      <div><p className="font-medium">Kelebihan</p><ul className="list-disc pl-5">{r.advantages.map((a) => <li key={a}>{a}</li>)}</ul></div>
                      <div><p className="font-medium">Tips penataan</p><ul className="list-disc pl-5">{r.styling_tips.map((t) => <li key={t}>{t}</li>)}</ul></div>
                    </div>
                    <details className="mt-3 text-sm">
                      <summary className="flex min-h-11 cursor-pointer items-center text-[var(--muted)]">Rincian skor</summary>
                      <ul className="mt-2 space-y-1">
                        {r.score_breakdown.map((t) => (
                          <li key={t.rule} className="flex gap-3"><span className="w-10 shrink-0 text-right tabular-nums">{t.points > 0 ? `+${t.points}` : t.points}</span><span>{t.reason}</span></li>
                        ))}
                      </ul>
                    </details>
                    {r.reference_image_note && <p className="mt-2 text-xs text-[var(--muted)]">{r.reference_image_note}</p>}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </section>
      </div>

      {saved.length > 0 && (
        <section className={`${card} mt-6`}>
          <h2 className="text-lg font-semibold">Data Anda di server</h2>
          <p className="mt-1 text-xs text-[var(--muted)]">Daftar ini hanya ada di peramban ini. Hapus analisis yang tidak Anda perlukan lagi.</p>
          {storageBlocked && (
            <p role="alert" className="mt-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900">
              Peramban ini tidak mengizinkan penyimpanan data situs, jadi daftar ini hilang jika halaman ditutup atau dimuat ulang.
              Hapus analisis yang tidak Anda perlukan sebelum menutup halaman.
            </p>
          )}
          <ul className="mt-3 divide-y divide-[var(--border)] text-sm">
            {saved.map((x) => (
              <li key={x.id} className="flex items-center justify-between gap-3 py-2">
                <span>
                  {new Date(x.createdAt).toLocaleString("id-ID")} · {x.imageStored ? "foto disimpan" : "tanpa foto"}
                  {analysis?.id === x.id && " · sedang ditampilkan"}
                </span>
                <button className={`${btnGhost} py-1.5 text-red-700`} onClick={() => remove(x.id, x.token, x.imageStored)} disabled={busy}>Hapus</button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <footer className="mt-12 border-t border-[var(--border)] pt-4 text-xs text-[var(--muted)]">
        Hasil bersifat perkiraan untuk riset dan bukan saran profesional. <a className="inline-block py-3 underline" href="/docs">Dokumentasi API</a>
      </footer>
    </main>
  );
}
