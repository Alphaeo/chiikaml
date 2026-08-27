import { useEffect, useRef, useState } from "react";

// Format des messages envoyes par le serveur C++ (SSE, sur /events).
// Volontairement minimal pour cette v1 : un seul "type" de payload
// (scatter), pense pour etre etendu plus tard (arbre de decision,
// courbes d'entrainement, etc.) via un registre de renderers gardes
// par `type`.
type ScatterPoint = { x: number; y: number; cluster: number };
type ScatterPayload = {
  type: "scatter";
  title: string;
  points: ScatterPoint[];
  centroids: { x: number; y: number }[];
};

const PALETTE = ["#4C9AFF", "#FF6B6B", "#51CF66", "#FFD43B", "#845EF7", "#20C997"];

function Scatter({ payload }: { payload: ScatterPayload }) {
  const width = 640;
  const height = 480;
  const padding = 40;

  const allX = [...payload.points.map((p) => p.x), ...payload.centroids.map((c) => c.x)];
  const allY = [...payload.points.map((p) => p.y), ...payload.centroids.map((c) => c.y)];
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const minY = Math.min(...allY);
  const maxY = Math.max(...allY);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const toSvgX = (x: number) => padding + ((x - minX) / spanX) * (width - 2 * padding);
  const toSvgY = (y: number) => height - padding - ((y - minY) / spanY) * (height - 2 * padding);

  return (
    <svg width={width} height={height} role="img" aria-label={payload.title}>
      <rect x={0} y={0} width={width} height={height} fill="var(--panel-bg)" />
      {payload.points.map((p, i) => (
        <circle
          key={i}
          cx={toSvgX(p.x)}
          cy={toSvgY(p.y)}
          r={5}
          fill={PALETTE[p.cluster % PALETTE.length]}
          opacity={0.85}
        />
      ))}
      {payload.centroids.map((c, i) => (
        <g key={i} stroke={PALETTE[i % PALETTE.length]} strokeWidth={2}>
          <line x1={toSvgX(c.x) - 8} y1={toSvgY(c.y) - 8} x2={toSvgX(c.x) + 8} y2={toSvgY(c.y) + 8} />
          <line x1={toSvgX(c.x) - 8} y1={toSvgY(c.y) + 8} x2={toSvgX(c.x) + 8} y2={toSvgY(c.y) - 8} />
        </g>
      ))}
    </svg>
  );
}

export default function App() {
  const [payload, setPayload] = useState<ScatterPayload | null>(null);
  const [connected, setConnected] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const source = new EventSource("/events");
    sourceRef.current = source;

    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (event) => {
      const data = JSON.parse(event.data) as ScatterPayload;
      setPayload(data);
    };

    return () => source.close();
  }, []);

  return (
    <main className="page">
      <header>
        <h1>chiikaml</h1>
        <span className={connected ? "status status-ok" : "status status-down"}>
          {connected ? "connecte" : "deconnecte"}
        </span>
      </header>

      {payload ? (
        <section>
          <h2>{payload.title}</h2>
          <Scatter payload={payload} />
        </section>
      ) : (
        <p className="waiting">En attente de donnees du programme C++...</p>
      )}
    </main>
  );
}
