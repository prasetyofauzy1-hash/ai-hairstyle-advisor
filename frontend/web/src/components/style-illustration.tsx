"use client";

import { useId } from "react";

// Hand-drawn front-view schematics, not photos: no license-clear reference photos exist for the catalog.
type Back = "none" | "ear" | "jaw" | "chin" | "shoulder" | "long" | "mullet";
type Sides = "shaved" | "tight" | "normal" | "full";
type Fringe = "none" | "full" | "short" | "side" | "curtain";
type Part = "none" | "side" | "center";
type Texture = "straight" | "wavy" | "curly" | "coily" | "braids" | "locs" | "twists";
type Extra = "blunt" | "aline" | "vcut" | "layers" | "asym" | "slick" | "bun" | "afro" | "hightop" | "jagged";
type Spec = { back: Back; sides: Sides; top: number; fringe: Fringe; part: Part; tex: Texture; extra?: Extra[] };

const s = (back: Back, sides: Sides, top: number, fringe: Fringe, part: Part, tex: Texture, ...extra: Extra[]): Spec =>
  ({ back, sides, top, fringe, part, tex, extra });

export const STYLE_SPECS: Record<string, Spec> = {
  pixie_cut: s("ear", "normal", 4, "none", "side", "straight"),
  pixie_side_bangs: s("ear", "normal", 4, "side", "none", "straight"),
  curly_pixie: s("ear", "tight", 8, "none", "none", "curly"),
  blunt_bob: s("jaw", "normal", 2, "none", "center", "straight", "blunt"),
  a_line_bob: s("chin", "normal", 2, "none", "center", "straight", "aline"),
  textured_wavy_bob: s("jaw", "full", 3, "none", "side", "wavy", "layers"),
  french_bob: s("jaw", "normal", 2, "full", "none", "straight", "blunt"),
  asymmetrical_bob: s("jaw", "normal", 2, "none", "side", "straight", "asym"),
  lob: s("shoulder", "normal", 2, "none", "side", "straight"),
  layered_lob: s("shoulder", "normal", 3, "none", "side", "wavy", "layers"),
  shag: s("shoulder", "full", 6, "side", "none", "wavy", "layers"),
  curtain_bangs_medium: s("shoulder", "normal", 3, "curtain", "center", "straight", "layers"),
  wolf_cut: s("shoulder", "full", 8, "none", "center", "wavy", "layers"),
  curly_shoulder_layers: s("shoulder", "full", 6, "none", "none", "curly", "layers"),
  medium_blunt: s("shoulder", "normal", 2, "none", "center", "straight", "blunt"),
  hush_cut: s("shoulder", "normal", 4, "curtain", "none", "wavy", "layers"),
  long_layers: s("long", "normal", 3, "none", "center", "straight", "layers"),
  sleek_long_straight: s("long", "normal", 1, "none", "center", "straight", "blunt"),
  long_waves_side_part: s("long", "full", 3, "none", "side", "wavy"),
  long_full_bangs: s("long", "normal", 2, "full", "none", "straight", "blunt"),
  butterfly_cut: s("long", "full", 5, "curtain", "center", "wavy", "layers"),
  long_curly_layers: s("long", "full", 6, "none", "none", "curly", "layers"),
  long_v_layers: s("long", "normal", 2, "none", "center", "straight", "vcut"),
  tapered_natural_curls: s("ear", "tight", 8, "none", "none", "coily"),
  buzz_cut: s("none", "shaved", 0, "none", "none", "straight"),
  crew_cut: s("none", "tight", 3, "none", "none", "straight"),
  ivy_league: s("none", "tight", 4, "none", "side", "straight"),
  classic_side_part: s("ear", "normal", 5, "none", "side", "straight"),
  pompadour: s("none", "tight", 14, "none", "none", "straight", "slick"),
  quiff: s("none", "tight", 11, "none", "none", "wavy"),
  french_crop: s("none", "tight", 4, "short", "none", "straight", "jagged"),
  undercut_slick_back: s("none", "shaved", 8, "none", "none", "straight", "slick"),
  taper_curly_top: s("none", "tight", 9, "none", "none", "curly"),
  two_block: s("ear", "normal", 5, "full", "none", "straight"),
  comma_hair: s("ear", "normal", 5, "side", "none", "straight"),
  curtain_hair_men: s("ear", "normal", 5, "curtain", "center", "straight"),
  messy_fringe: s("ear", "normal", 5, "full", "none", "straight", "jagged"),
  caesar_cut: s("none", "tight", 2, "short", "none", "straight"),
  medium_slick_back: s("jaw", "normal", 6, "none", "none", "straight", "slick"),
  man_bun: s("none", "normal", 2, "none", "none", "straight", "slick", "bun"),
  bro_flow: s("shoulder", "normal", 5, "none", "none", "wavy", "slick"),
  modern_mullet: s("mullet", "tight", 6, "short", "none", "wavy", "jagged"),
  short_afro: s("none", "full", 8, "none", "none", "coily", "afro"),
  high_top_fade: s("none", "tight", 0, "none", "none", "coily", "hightop"),
  short_twists: s("ear", "normal", 8, "none", "none", "twists"),
  side_swept_taper: s("none", "tight", 5, "side", "side", "straight"),
  curly_fringe: s("none", "tight", 6, "full", "none", "curly", "jagged"),
  long_layered_men: s("shoulder", "full", 4, "none", "center", "wavy", "layers"),
  afro_rounded_medium: s("none", "full", 16, "none", "none", "coily", "afro"),
  twist_out: s("shoulder", "full", 8, "none", "none", "coily", "layers"),
  box_braids_side_part: s("long", "normal", 2, "none", "side", "braids"),
  tapered_cut_side_bangs: s("ear", "tight", 5, "side", "none", "coily"),
  afro_taper_part: s("none", "tight", 7, "none", "side", "coily"),
  twist_fringe: s("none", "tight", 7, "full", "none", "twists"),
  long_locs: s("long", "full", 4, "none", "none", "locs"),
};

const HAIR = "#3b302b";
const FADE = "#a59b94";
const LINE = "#8a7d75";
const BOTTOM: Record<Back, number> = { none: 0, ear: 0, jaw: 100, chin: 106, shoulder: 118, long: 136, mullet: 118 };
const SIDE_W: Record<Sides, number> = { shaved: 0, tight: 0, normal: 2, full: 6 };

function backPath(sp: Spec, sideW: number): string | null {
  const ex = sp.extra ?? [];
  if (sp.back === "mullet") return "M43 66 L42 118 L78 118 L77 66 Z";
  const yb = BOTTOM[sp.back];
  if (!yb) return null;
  const L = 33 - sideW, R = 87 + sideW;
  const out = sp.back === "long" ? 6 : sp.back === "shoulder" ? 4 : 1;
  const Lb = L - out, Rb = R + out;
  if (ex.includes("asym"))
    return `M${L} 64 L${Lb} 98 L50 100 L70 100 L${Rb} 112 L${R} 64 Z`;
  if (ex.includes("aline"))
    return `M${L} 64 L${Lb} ${yb - 10} L50 ${yb + 2} L70 ${yb + 2} L${Rb} ${yb - 10} L${R} 64 Z`;
  if (ex.includes("vcut"))
    return `M${L} 64 L${Lb} ${yb} L50 ${yb - 14} L70 ${yb - 14} L${Rb} ${yb} L${R} 64 Z`;
  if (ex.includes("layers")) {
    const scallops = (from: number, to: number) => {
      const step = (to - from) / 3;
      return [0, 1, 2].map((i) => `Q${from + step * (i + 0.5)} ${yb + 5} ${from + step * (i + 1)} ${yb - (i === 2 ? 0 : 3)}`).join(" ");
    };
    return `M${L} 64 L${Lb} ${yb - 3} ${scallops(Lb, 50)} L70 ${yb} ${scallops(70, Rb)} L${R} 64 Z`;
  }
  return `M${L} 64 L${Lb} ${yb} L${Rb} ${yb} L${R} 64 Z`;
}

function fringePath(f: Fringe, jagged: boolean): string[] {
  const zig = (y: number) => `L84 ${y} L78 ${y - 3} L72 ${y + 1} L66 ${y - 3} L60 ${y + 1} L54 ${y - 3} L48 ${y + 1} L42 ${y - 3} L36 ${y} Z`;
  switch (f) {
    case "full": return [jagged ? `M36 52 Q60 36 84 52 ${zig(58)}` : "M36 52 Q60 36 84 52 L84 59 L36 59 Z"];
    case "short": return [`M37 50 Q60 36 83 50 ${zig(51).replace("L84", "L83").replace("L36", "L37")}`];
    case "side": return ["M36 56 Q50 36 84 50 L84 55 Q62 52 39 65 Z"];
    case "curtain": return ["M60 44 Q44 46 36 64 L40 66 Q46 52 60 50 Z", "M60 44 Q76 46 84 64 L80 66 Q74 52 60 50 Z"];
    default: return [];
  }
}

export function StyleIllustration({ id, name, className }: { id: string; name: string; className?: string }) {
  const uid = useId().replace(/:/g, "");
  const sp = STYLE_SPECS[id];
  if (!sp) return null;
  const ex = sp.extra ?? [];
  const sideW = SIDE_W[sp.sides];
  const topY = 30 - sp.top;
  const xl = 34 - sideW, xr = 86 + sideW;
  const faded = sp.sides === "tight" || sp.sides === "shaved";
  const capPath = (h: number) => `M${xl} 72 A${26 + sideW} ${42 + h} 0 0 1 ${xr} 72 L84 60 Q60 34 36 60 Z`;
  const cap = capPath(sp.top);
  const top = ex.includes("hightop")
    ? "M37 58 L39 12 L81 12 L83 58 Q60 34 37 58 Z"
    : `M37 58 A23 ${28 + sp.top} 0 0 1 83 58 Q60 34 37 58 Z`;
  const hair: string[] = [];
  const back = backPath(sp, sideW);
  if (ex.includes("afro")) hair.push(`M60 ${topY - 4} a${30 + sp.top} ${32 + sp.top / 2} 0 1 0 0.1 0 Z`);
  if (back) hair.push(back);
  const front: string[] = [];
  if (ex.includes("bun")) front.push(`M60 ${topY - 18} a10 10 0 1 0 0.1 0 Z`);
  if (sp.sides !== "shaved" || sp.top > 0) front.push(faded ? top : cap);
  front.push(...fringePath(sp.fringe, ex.includes("jagged")));
  const tex = `${uid}-tex`;
  const showEars = !back || sp.back === "mullet";
  const strands = ex.includes("slick")
    ? [44, 52, 60, 68, 76].map((x) => `M${x} 48 Q${x + (x - 60) * 0.2} ${40 - sp.top / 2} ${x + (x - 60) * 0.1} ${topY + 8}`)
    : [];
  const partX = sp.part === "side" ? 47 : 60;
  return (
    <svg viewBox="0 0 120 140" role="img" aria-label={`Ilustrasi ${name}`} className={className}>
      <title>{`Ilustrasi ${name}`}</title>
      <defs>
        <pattern id={tex} width="8" height="8" patternUnits="userSpaceOnUse">
          {sp.tex === "straight" && <path d="M2 0 V8 M6 0 V8" stroke={LINE} strokeWidth="0.4" />}
          {sp.tex === "wavy" && <path d="M0 4 Q2 1 4 4 T8 4" fill="none" stroke={LINE} strokeWidth="0.7" />}
          {sp.tex === "curly" && <circle cx="4" cy="4" r="2.4" fill="none" stroke={LINE} strokeWidth="0.7" />}
          {sp.tex === "coily" && <path d="M2 2 m-1.3 0 a1.3 1.3 0 1 0 2.6 0 a1.3 1.3 0 1 0 -2.6 0 M6 6 m-1.3 0 a1.3 1.3 0 1 0 2.6 0 a1.3 1.3 0 1 0 -2.6 0" fill="none" stroke={LINE} strokeWidth="0.6" />}
          {sp.tex === "braids" && <path d="M0 0 L4 3 L8 0 M0 4 L4 7 L8 4" fill="none" stroke={LINE} strokeWidth="0.7" />}
          {sp.tex === "locs" && <path d="M2 0 V7 M6 1 V8" stroke={LINE} strokeWidth="1.6" strokeLinecap="round" />}
          {sp.tex === "twists" && <path d="M1 7 L3 1 M5 7 L7 1" stroke={LINE} strokeWidth="0.9" strokeLinecap="round" />}
        </pattern>
      </defs>
      <path d="M16 140 C18 122 38 116 51 113 L69 113 C82 116 102 122 104 140 Z" fill="#dcd5cf" />
      {hair.map((d) => <g key={d}><path d={d} fill={HAIR} /><path d={d} fill={`url(#${tex})`} /></g>)}
      <path d="M51 92 L51 116 Q60 120 69 116 L69 92 Z" fill="#e7ddd4" />
      {showEars && <><ellipse cx="34" cy="73" rx="4" ry="7" fill="#e7ddd4" /><ellipse cx="86" cy="73" rx="4" ry="7" fill="#e7ddd4" /></>}
      <ellipse cx="60" cy="70" rx="25" ry="32" fill="#efe6de" stroke="#cbbfb5" strokeWidth="0.8" />
      {faded && <path d={capPath(Math.min(sp.top, 3))} fill={FADE} opacity="0.85" />}
      {front.map((d) => <g key={d}><path d={d} fill={HAIR} /><path d={d} fill={`url(#${tex})`} /></g>)}
      {strands.map((d) => <path key={d} d={d} fill="none" stroke={LINE} strokeWidth="0.8" />)}
      {sp.part !== "none" && <path d={`M${partX} ${topY + (sp.part === "side" ? 10 : 5)} Q${partX + 1} 40 ${partX} 47`} fill="none" stroke={sp.tex === "coily" ? FADE : "#1d1715"} strokeWidth="1.2" />}
      <path d="M45 66 Q50 64 55 66 M65 66 Q70 64 75 66" fill="none" stroke="#9a8c82" strokeWidth="1" />
      <ellipse cx="50" cy="73" rx="2.2" ry="1.4" fill="#6b6360" />
      <ellipse cx="70" cy="73" rx="2.2" ry="1.4" fill="#6b6360" />
      <path d="M60 76 L58 84 L61 85" fill="none" stroke="#b9aca2" strokeWidth="0.9" />
      <path d="M54 91 Q60 94 66 91" fill="none" stroke="#b08f86" strokeWidth="1.1" />
    </svg>
  );
}
