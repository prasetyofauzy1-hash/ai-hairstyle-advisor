"use client";

import { useEffect, useState } from "react";
import {
  Analysis, HAIR_TYPES, LENGTHS, MAINT, RecRequest, Recs, SHAPES, THICKNESS,
  createAnalysis, deleteAnalysis, getRecommendations,
} from "@/lib/api";

const card = "rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 sm:p-6 shadow-sm";
const btn = "rounded-lg bg-[var(--accent)] px-4 py-2.5 text-sm font-medium text-[var(--accent-fg)] disabled:opacity-40";
const btnGhost = "rounded-lg border border-[var(--border)] px-4 py-2.5 text-sm font-medium";
const label = "block text-sm font-medium mb-1";
const select = "w-full rounded-lg border border-[var(--border)] bg-white px-3 py-2 text-sm";

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
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [recs, setRecs] = useState<Recs | null>(null);
  const [deleted, setDeleted] = useState(false);
  const [prefs, setPrefs] = useState<RecRequest>({
    hair_type: "straight", thickness: "medium", gender: "", preferred_length: "", max_maintenance: "",
    avoid_bangs: false, formal: false, face_shape: "",
  });

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  function pick(f: File | null) {
    setFile(f); setAnalysis(null); setRecs(null); setError(null); setDeleted(false);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  async function analyze() {
    if (!file) return;
    setBusy(true); setError(null); setRecs(null);
    try {
      const a = await createAnalysis(file, consent);
      setAnalysis(a); setToken(a.access_token ?? null);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function recommend() {
    if (!analysis || !token) return;
    setBusy(true); setError(null);
    const clean = Object.fromEntries(Object.entries(prefs).map(([k, v]) => [k, v === "" ? null : v])) as RecRequest;
    try { setRecs(await getRecommendations(analysis.id, token, clean)); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function remove() {
    if (!analysis || !token) return;
    setBusy(true);
    try {
      await deleteAnalysis(analysis.id, token);
      pick(null); setToken(null); setDeleted(true);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  const set = (k: keyof RecRequest) => (v: string | boolean) => setPrefs((p) => ({ ...p, [k]: v }));
  const pred = analysis?.prediction;
  const canRecommend = !!pred || !!prefs.face_shape;

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 sm:py-12">
      <header className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-widest text-[var(--accent)]">Proyek riset · non-komersial</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight sm:text-4xl">AI Hairstyle Advisor</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Unggah foto wajah tampak depan. Sistem mengukur proporsi wajah, memperkirakan bentuk wajah,
          lalu merekomendasikan gaya rambut beserta alasannya.
        </p>
      </header>

      {deleted && (
        <div className="mb-6 rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-900">
          Analisis, foto, dan rekomendasi Anda sudah dihapus dari server.
        </div>
      )}
      {error && (
        <div role="alert" className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">{error}</div>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <section className={card}>
          <h2 className="text-lg font-semibold">1. Unggah foto</h2>
          <label className="mt-4 flex aspect-[4/5] cursor-pointer items-center justify-center overflow-hidden rounded-lg border-2 border-dashed border-[var(--border)] bg-stone-50">
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="Pratinjau foto" className="h-full w-full object-cover" />
            ) : (
              <span className="px-6 text-center text-sm text-[var(--muted)]">Klik untuk memilih foto (JPEG, PNG, atau WebP, maks. 8 MB)</span>
            )}
            <input type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" aria-label="Pilih foto"
              onChange={(e) => pick(e.target.files?.[0] ?? null)} />
          </label>
          <p className="mt-3 text-xs text-[var(--muted)]">Tips: wajah menghadap kamera, cahaya merata, tanpa kacamata hitam, dahi terlihat.</p>

          <label className="mt-4 flex items-start gap-3 rounded-lg bg-stone-50 p-3 text-sm">
            <input type="checkbox" className="mt-0.5 h-4 w-4" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            <span>
              Saya setuju foto saya <strong>disimpan</strong> di server untuk riwayat analisis. Tanpa centang ini, foto hanya
              diproses di memori dan tidak disimpan. Foto bisa dihapus kapan saja.
            </span>
          </label>

          <button className={`${btn} mt-4 w-full`} disabled={!file || busy} onClick={analyze}>
            {busy && !analysis ? "Menganalisis..." : "Analisis wajah"}
          </button>
        </section>

        <section className="space-y-6">
          {!analysis && (
            <div className={`${card} text-sm text-[var(--muted)]`}>Hasil analisis akan muncul di sini.</div>
          )}

          {analysis && (
            <div className={card}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <h2 className="text-lg font-semibold">2. Hasil analisis</h2>
                <button className={`${btnGhost} text-red-700`} onClick={remove} disabled={busy}>Hapus analisis & foto</button>
              </div>
              <p className="mt-1 text-xs text-[var(--muted)]">
                {analysis.image_stored ? "Foto disimpan atas persetujuan Anda." : "Foto tidak disimpan."}
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
                <p className="mt-4 text-sm">Foto belum memenuhi syarat kualitas, jadi bentuk wajah tidak diperkirakan. Coba foto lain, atau pilih bentuk wajah secara manual di bawah.</p>
              )}

              {pred && (
                <div className="mt-4">
                  <p className="text-sm text-[var(--muted)]">Perkiraan bentuk wajah</p>
                  <p className="text-3xl font-semibold">{SHAPES[pred.shape] ?? pred.label_id}</p>
                  <div className="mt-4 space-y-1.5">
                    {Object.entries(pred.scores).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                      <div key={k} className="flex items-center gap-3 text-sm">
                        <span className="w-20 shrink-0">{SHAPES[k] ?? k}</span>
                        <div className="h-2 flex-1 rounded bg-stone-100">
                          <div className="h-2 rounded bg-[var(--accent)]" style={{ width: `${v * 100}%` }} />
                        </div>
                        <span className="w-12 text-right tabular-nums text-[var(--muted)]">{v.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-[var(--muted)]">
                    Skor keluaran model ({pred.method}), <strong>tidak terkalibrasi</strong>: bukan peluang benar.
                    {analysis.rule_baseline && <> Pembanding aturan Fase 2: {SHAPES[analysis.rule_baseline.shape] ?? analysis.rule_baseline.shape}.</>}
                  </p>
                </div>
              )}

              <details className="mt-4 rounded-lg bg-amber-50 p-3 text-sm text-amber-950" open>
                <summary className="cursor-pointer font-medium">Keterbatasan model</summary>
                <ul className="mt-2 list-disc space-y-1 pl-5">{analysis.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
              </details>
            </div>
          )}

          {analysis && (
            <div className={card}>
              <h2 className="text-lg font-semibold">3. Profil rambut</h2>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <Select name="Jenis rambut" value={prefs.hair_type} onChange={set("hair_type")} options={HAIR_TYPES} />
                <Select name="Ketebalan" value={prefs.thickness} onChange={set("thickness")} options={THICKNESS} />
                <Select name="Kategori gaya" value={prefs.gender ?? ""} onChange={set("gender")} options={{ pria: "Pria", wanita: "Wanita" }} empty="Semua" />
                <Select name="Panjang yang diinginkan" value={prefs.preferred_length ?? ""} onChange={set("preferred_length")} options={LENGTHS} empty="Bebas" />
                <Select name="Perawatan maksimal" value={prefs.max_maintenance ?? ""} onChange={set("max_maintenance")} options={MAINT} empty="Bebas" />
                <Select name="Bentuk wajah" value={prefs.face_shape ?? ""} onChange={set("face_shape")} options={SHAPES}
                  empty={pred ? `Ikuti hasil model (${SHAPES[pred.shape]})` : "Pilih bentuk wajah"} />
              </div>
              <div className="mt-4 flex flex-wrap gap-5 text-sm">
                <label className="flex items-center gap-2"><input type="checkbox" checked={!!prefs.avoid_bangs} onChange={(e) => set("avoid_bangs")(e.target.checked)} /> Hindari poni</label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={!!prefs.formal} onChange={(e) => set("formal")(e.target.checked)} /> Untuk acara formal / kantor</label>
              </div>
              <button className={`${btn} mt-5`} onClick={recommend} disabled={busy || !canRecommend}>
                {busy ? "Memproses..." : "Tampilkan rekomendasi"}
              </button>
            </div>
          )}

          {recs && (
            <div className={card}>
              <h2 className="text-lg font-semibold">4. Rekomendasi untuk wajah {recs.face_shape_label}</h2>
              <p className="mt-1 text-xs text-[var(--muted)]">
                Bentuk wajah dari {recs.face_shape_source === "model" ? "hasil model" : "pilihan Anda"}.
              </p>
              {recs.notes.length > 0 && (
                <ul className="mt-3 space-y-1 text-sm text-amber-900">{recs.notes.map((n) => <li key={n}>• {n}</li>)}</ul>
              )}
              {recs.recommendations.length === 0 && <p className="mt-4 text-sm">Tidak ada gaya yang cocok dengan filter ini. Longgarkan preferensi Anda.</p>}
              <ol className="mt-4 space-y-4">
                {recs.recommendations.map((r) => (
                  <li key={r.id} className="rounded-lg border border-[var(--border)] p-4">
                    <div className="flex items-baseline justify-between gap-3">
                      <h3 className="font-semibold">{r.rank}. {r.name}</h3>
                      <span className="shrink-0 rounded bg-stone-100 px-2 py-0.5 text-xs tabular-nums">{r.score} {r.score_unit}</span>
                    </div>
                    <p className="mt-1 text-xs text-[var(--muted)]">Panjang: {r.length} · Perawatan: {r.maintenance}</p>
                    <p className="mt-2 text-sm">{r.description}</p>
                    <p className="mt-2 text-sm"><strong>Mengapa cocok:</strong> {r.explanation}</p>
                    <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
                      <div><p className="font-medium">Kelebihan</p><ul className="list-disc pl-5">{r.advantages.map((a) => <li key={a}>{a}</li>)}</ul></div>
                      <div><p className="font-medium">Tips penataan</p><ul className="list-disc pl-5">{r.styling_tips.map((t) => <li key={t}>{t}</li>)}</ul></div>
                    </div>
                    <details className="mt-3 text-sm">
                      <summary className="cursor-pointer text-[var(--muted)]">Rincian skor</summary>
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

      <footer className="mt-12 border-t border-[var(--border)] pt-4 text-xs text-[var(--muted)]">
        Hasil bersifat perkiraan untuk riset dan bukan saran profesional. <a className="underline" href="/docs">Dokumentasi API</a>
      </footer>
    </main>
  );
}
