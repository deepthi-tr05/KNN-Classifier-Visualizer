import { useState, useMemo, useCallback } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { irisData, irisTargets, irisTargetNames, irisFeatureNames } from "./data/irisData";
import { computePCA } from "./utils/pca";
import { computeFindSSteps, ATTRIBUTES } from "./data/findsData";

// ── KNN core ────────────────────────────────────────────────────────────────

function euclidean(a: number[], b: number[]): number {
  return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
}

function knnPredict(
  trainX: number[][],
  trainY: number[],
  query: number[],
  k: number
): { label: number; neighbors: { idx: number; dist: number }[] } {
  const dists = trainX.map((x, idx) => ({ idx, dist: euclidean(x, query) }));
  dists.sort((a, b) => a.dist - b.dist);
  const neighbors = dists.slice(0, k);
  const votes: Record<number, number> = {};
  for (const n of neighbors) {
    votes[trainY[n.idx]] = (votes[trainY[n.idx]] ?? 0) + 1;
  }
  const label = parseInt(
    Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]
  );
  return { label, neighbors };
}

// ── Colour palette ───────────────────────────────────────────────────────────

const CLASS_COLORS = ["#6ee7f7", "#f97316", "#a78bfa"];
const CLASS_COLORS_DARK = ["#0891b2", "#c2410c", "#7c3aed"];
const NEIGHBOR_COLOR = "#facc15";

// ── Types ────────────────────────────────────────────────────────────────────

type Tab = "knn" | "finds";

// ── Custom tooltip ────────────────────────────────────────────────────────────

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: { name: string; x: number; y: number; label: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-xs shadow-xl">
      <p className="font-semibold text-white">{d.name}</p>
      <p className="text-gray-400">
        PC1: <span className="text-cyan-300">{d.x.toFixed(3)}</span>
      </p>
      <p className="text-gray-400">
        PC2: <span className="text-cyan-300">{d.y.toFixed(3)}</span>
      </p>
    </div>
  );
}

// ── KNN Tab ───────────────────────────────────────────────────────────────────

function KNNTab() {
  const [k, setK] = useState(3);
  const [queryFeature, setQueryFeature] = useState([5.8, 3.0, 4.5, 1.5]);
  const [activeFeature, setActiveFeature] = useState<number | null>(null);

  // PCA once
  const pca = useMemo(() => computePCA(irisData, 2), []);

  const points = useMemo(
    () =>
      pca.transformed.map((pt, i) => ({
        x: pt[0],
        y: pt[1],
        label: irisTargets[i],
        name: irisTargetNames[irisTargets[i]],
      })),
    [pca]
  );

  const queryPCA = useMemo(() => {
    const centered = queryFeature.map((v, i) => v - pca.means[i]);
    return pca.components.map((comp) =>
      comp.reduce((s, c, i) => s + c * centered[i], 0)
    );
  }, [queryFeature, pca]);

  const result = useMemo(
    () => knnPredict(irisData, irisTargets, queryFeature, k),
    [queryFeature, k]
  );

  const neighborSet = useMemo(
    () => new Set(result.neighbors.map((n) => n.idx)),
    [result]
  );

  // Split scatter data by class + highlight neighbors
  const scatterSeries = useMemo(() => {
    const byClass: Record<number, { x: number; y: number; name: string; label: number }[]> = {
      0: [], 1: [], 2: [],
    };
    points.forEach((p, i) => {
      if (!neighborSet.has(i)) byClass[p.label].push(p);
    });
    return byClass;
  }, [points, neighborSet]);

  const neighborPoints = useMemo(
    () =>
      result.neighbors.map((n) => ({
        x: pca.transformed[n.idx][0],
        y: pca.transformed[n.idx][1],
        name: irisTargetNames[irisTargets[n.idx]],
        label: irisTargets[n.idx],
      })),
    [result, pca]
  );

  const handleSlider = useCallback(
    (i: number, val: number) => {
      setQueryFeature((prev) => {
        const next = [...prev];
        next[i] = val;
        return next;
      });
    },
    []
  );

  const featureRanges = [
    { min: 4.0, max: 8.0, step: 0.1 },
    { min: 2.0, max: 4.5, step: 0.1 },
    { min: 1.0, max: 7.0, step: 0.1 },
    { min: 0.1, max: 2.5, step: 0.1 },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* K slider */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-semibold text-cyan-300 mb-3 uppercase tracking-widest">
            Neighbours (K)
          </h3>
          <div className="flex items-center gap-4">
            <input
              id="k-slider"
              type="range"
              min={1}
              max={15}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="slider flex-1"
            />
            <span className="text-3xl font-black text-white w-10 text-center">
              {k}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Larger K → smoother boundaries. Odd K avoids ties.
          </p>
        </div>

        {/* Prediction result */}
        <div
          className="glass-card p-5 flex flex-col justify-between"
          style={{ borderColor: CLASS_COLORS[result.label] + "55" }}
        >
          <h3 className="text-sm font-semibold text-cyan-300 mb-1 uppercase tracking-widest">
            Prediction
          </h3>
          <div className="flex items-center gap-3 mt-2">
            <div
              className="w-4 h-4 rounded-full flex-shrink-0"
              style={{ background: CLASS_COLORS[result.label] }}
            />
            <span
              className="text-2xl font-black"
              style={{ color: CLASS_COLORS[result.label] }}
            >
              {irisTargetNames[result.label]}
            </span>
          </div>
          <div className="flex gap-2 mt-3 flex-wrap">
            {result.neighbors.map((n, i) => (
              <span
                key={i}
                className="text-xs px-2 py-1 rounded-full border"
                style={{
                  borderColor: CLASS_COLORS[irisTargets[n.idx]] + "88",
                  color: CLASS_COLORS[irisTargets[n.idx]],
                  background: CLASS_COLORS[irisTargets[n.idx]] + "18",
                }}
              >
                {irisTargetNames[irisTargets[n.idx]]} (d={n.dist.toFixed(2)})
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Feature sliders */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-cyan-300 mb-4 uppercase tracking-widest">
          Query Point Features
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {irisFeatureNames.map((name, i) => (
            <div
              key={i}
              className={`rounded-xl p-3 transition-all duration-200 ${
                activeFeature === i
                  ? "bg-white/5 ring-1 ring-cyan-500/40"
                  : "bg-white/[0.02]"
              }`}
              onMouseEnter={() => setActiveFeature(i)}
              onMouseLeave={() => setActiveFeature(null)}
            >
              <div className="flex justify-between mb-1">
                <span className="text-xs text-gray-400">{name}</span>
                <span className="text-xs font-mono text-cyan-300">
                  {queryFeature[i].toFixed(1)} cm
                </span>
              </div>
              <input
                id={`feature-slider-${i}`}
                type="range"
                min={featureRanges[i].min}
                max={featureRanges[i].max}
                step={featureRanges[i].step}
                value={queryFeature[i]}
                onChange={(e) => handleSlider(i, Number(e.target.value))}
                className="slider w-full"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Scatter plot */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-cyan-300 mb-1 uppercase tracking-widest">
          PCA Projection (2D)
        </h3>
        <p className="text-xs text-gray-500 mb-4">
          Variance explained: PC1{" "}
          {(pca.explainedVarianceRatio[0] * 100).toFixed(1)}% · PC2{" "}
          {(pca.explainedVarianceRatio[1] * 100).toFixed(1)}%
        </p>
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
            <XAxis
              dataKey="x"
              type="number"
              name="PC1"
              tick={{ fill: "#6b7280", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#374151" }}
              domain={["auto", "auto"]}
            />
            <YAxis
              dataKey="y"
              type="number"
              name="PC2"
              tick={{ fill: "#6b7280", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#374151" }}
              domain={["auto", "auto"]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: "12px", paddingTop: "12px" }}
            />
            {[0, 1, 2].map((cls) => (
              <Scatter
                key={cls}
                name={irisTargetNames[cls]}
                data={scatterSeries[cls]}
                fill={CLASS_COLORS[cls]}
                opacity={0.75}
                r={5}
              />
            ))}
            <Scatter
              name="Neighbours"
              data={neighborPoints}
              fill={NEIGHBOR_COLOR}
              opacity={1}
              r={7}
              shape={(props: {cx: number; cy: number}) => (
                <circle
                  cx={props.cx}
                  cy={props.cy}
                  r={8}
                  fill={NEIGHBOR_COLOR}
                  stroke="#fff"
                  strokeWidth={2}
                />
              )}
            />
            <Scatter
              name="Query"
              data={[{ x: queryPCA[0], y: queryPCA[1], name: "Query", label: result.label }]}
              fill="#ffffff"
              r={10}
              shape={(props: {cx: number; cy: number}) => (
                <g>
                  <circle cx={props.cx} cy={props.cy} r={10} fill="#fff" stroke={CLASS_COLORS[result.label]} strokeWidth={3} />
                  <text x={props.cx} y={props.cy + 4} textAnchor="middle" fontSize={10} fontWeight="bold" fill={CLASS_COLORS_DARK[result.label]}>Q</text>
                </g>
              )}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ── FIND-S Tab ────────────────────────────────────────────────────────────────

function FindSTab() {
  const steps = useMemo(() => computeFindSSteps(), []);
  const [currentStep, setCurrentStep] = useState(0);

  const step = steps[currentStep];

  const eventLabel: Record<string, { label: string; color: string }> = {
    init: { label: "Initialized", color: "text-cyan-400" },
    "positive-match": { label: "Positive Match", color: "text-green-400" },
    "positive-mismatch": { label: "Generalized", color: "text-orange-400" },
    skipped: { label: "Skipped (Negative)", color: "text-gray-500" },
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-cyan-300 mb-1 uppercase tracking-widest">
          FIND-S Algorithm — Enjoy Sport Dataset
        </h3>
        <p className="text-xs text-gray-500 mb-4">
          Step through the FIND-S algorithm. It initialises the hypothesis from
          the first positive example and generalises whenever a positive example
          doesn't match.
        </p>

        {/* Step navigation */}
        <div className="flex items-center gap-3 mb-6">
          <button
            id="finds-prev"
            onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
            disabled={currentStep === 0}
            className="btn-secondary"
          >
            ← Prev
          </button>
          <div className="flex gap-1 flex-1 justify-center">
            {steps.map((_, i) => (
              <button
                key={i}
                id={`finds-step-${i}`}
                onClick={() => setCurrentStep(i)}
                className={`w-7 h-7 rounded-full text-xs font-bold transition-all duration-200 ${
                  i === currentStep
                    ? "bg-cyan-500 text-black scale-110"
                    : "bg-white/10 text-gray-400 hover:bg-white/20"
                }`}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <button
            id="finds-next"
            onClick={() =>
              setCurrentStep((s) => Math.min(steps.length - 1, s + 1))
            }
            disabled={currentStep === steps.length - 1}
            className="btn-secondary"
          >
            Next →
          </button>
        </div>

        {/* Event badge */}
        <div className="mb-4">
          <span
            className={`text-xs font-semibold px-3 py-1 rounded-full border border-current ${
              eventLabel[step.event].color
            }`}
          >
            Example {step.step + 1} — {eventLabel[step.event].label}
          </span>
        </div>

        {/* Attribute table */}
        <div className="overflow-x-auto rounded-xl border border-white/10">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-white/5">
                <th className="px-4 py-2 text-left text-gray-400 font-medium text-xs uppercase tracking-wider">
                  Attribute
                </th>
                <th className="px-4 py-2 text-left text-gray-400 font-medium text-xs uppercase tracking-wider">
                  Example Value
                </th>
                <th className="px-4 py-2 text-left text-gray-400 font-medium text-xs uppercase tracking-wider">
                  Hypothesis
                </th>
              </tr>
            </thead>
            <tbody>
              {ATTRIBUTES.map((attr, i) => (
                <tr
                  key={attr}
                  className={`border-t border-white/5 transition-colors duration-200 ${
                    step.changedAttrs[i] ? "bg-cyan-500/10" : ""
                  }`}
                >
                  <td className="px-4 py-2 text-gray-300 font-medium">{attr}</td>
                  <td className="px-4 py-2 text-white font-mono">
                    {step.exampleValues[i]}
                  </td>
                  <td className="px-4 py-2">
                    <span
                      className={`font-mono font-bold ${
                        step.hypothesis[i] === "?"
                          ? "text-orange-400"
                          : step.hypothesis[i] === "∅"
                          ? "text-gray-600"
                          : step.changedAttrs[i]
                          ? "text-cyan-300"
                          : "text-white"
                      }`}
                    >
                      {step.hypothesis[i]}
                    </span>
                    {step.changedAttrs[i] && (
                      <span className="ml-2 text-xs text-cyan-500 animate-pulse">
                        ← updated
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Current hypothesis summary */}
        <div className="mt-4 p-3 rounded-xl bg-white/[0.03] border border-white/10">
          <p className="text-xs text-gray-500 mb-1">Current Hypothesis H:</p>
          <p className="text-sm font-mono text-cyan-300 break-all">
            {"< "}
            {step.hypothesis.join(", ")}
            {" >"}
          </p>
        </div>
      </div>

      {/* Algorithm explanation */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-cyan-300 mb-3 uppercase tracking-widest">
          How FIND-S Works
        </h3>
        <ol className="space-y-2 text-sm text-gray-400 list-decimal list-inside">
          <li>Start with the most specific hypothesis: <span className="text-gray-300 font-mono">∅</span></li>
          <li>For each <span className="text-green-400">positive</span> training example:</li>
          <li className="ml-4">If current H covers example → no change</li>
          <li className="ml-4">Otherwise → generalise each mismatching attribute to <span className="text-orange-400 font-mono">?</span></li>
          <li>Ignore all negative examples</li>
          <li>Return final hypothesis</li>
        </ol>
      </div>
    </div>
  );
}

// ── App root ──────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState<Tab>("knn");

  return (
    <div className="min-h-screen bg-[#0a0f1a] text-white font-sans">
      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-white/10 backdrop-blur-xl bg-[#0a0f1a]/80">
        <div className="max-w-5xl mx-auto px-4 py-4 flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-violet-600 flex items-center justify-center text-lg font-black select-none">
              K
            </div>
            <div>
              <h1 className="text-lg font-black tracking-tight leading-none">
                KNN Classifier Visualizer
              </h1>
              <p className="text-xs text-gray-500 mt-0.5">
                Interactive ML Explorer · Iris + FIND-S
              </p>
            </div>
          </div>
          <div className="sm:ml-auto flex gap-1 bg-white/5 rounded-xl p-1">
            <button
              id="tab-knn"
              onClick={() => setTab("knn")}
              className={`tab-btn ${tab === "knn" ? "tab-active" : ""}`}
            >
              🔵 KNN
            </button>
            <button
              id="tab-finds"
              onClick={() => setTab("finds")}
              className={`tab-btn ${tab === "finds" ? "tab-active" : ""}`}
            >
              🧠 FIND-S
            </button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-5xl mx-auto px-4 py-8">
        {tab === "knn" ? <KNNTab /> : <FindSTab />}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 mt-12 py-6 text-center">
        <p className="text-xs text-gray-600">
          Built with React · TypeScript · Vite · Recharts · Tailwind CSS
        </p>
      </footer>
    </div>
  );
}
