import { useMemo, useRef, useState, useLayoutEffect } from "react";
import Tree from "react-d3-tree";

export type TreeNode = {
  name: string;
  attributes?: Record<string, string | number | boolean>;
  children?: TreeNode[];
  status?: "accepted" | "rejected" | "pending";
};

type Props = {
  data: TreeNode;
  tokensPerSecond: number;
  speedup?: number | null;
};

const MAX_LABEL_CHARS = 9;
const LABEL_FONT_SIZE = 13;
const LABEL_FONT_WEIGHT = 600;
const LABEL_STROKE = "#0b1220";
const LABEL_STROKE_WIDTH = 3;

const statusColor = (status?: string) => {
  if (status === "accepted") return "#22c55e";
  if (status === "rejected") return "#ef4444";
  return "#94a3b8";
};

const truncateLabel = (label: string) => {
  if (label.length <= MAX_LABEL_CHARS) {
    return { displayLabel: label, truncated: false };
  }
  return {
    displayLabel: `${label.slice(0, MAX_LABEL_CHARS)}...`,
    truncated: true,
  };
};

export default function TreeViz({ data, tokensPerSecond, speedup }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const resize = () => {
      setDims({ width: el.clientWidth, height: el.clientHeight });
    };
    resize();

    const observer = new ResizeObserver(resize);
    observer.observe(el);

    return () => observer.disconnect();
  }, []);

  const translate = useMemo(
    () => ({ x: 80, y: dims.height / 2 }),
    [dims.height]
  );

  return (
    <div className="tree-container" ref={containerRef}>
      {dims.width > 0 && (
        <Tree
          data={data}
          orientation="horizontal"
          translate={translate}
          nodeSize={{ x: 140, y: 80 }}
          separation={{ siblings: 1.2, nonSiblings: 1.6 }}
          renderCustomNodeElement={({ nodeDatum }) => {
            const node = nodeDatum as TreeNode;
            const label = node.name ?? "";
            const { displayLabel, truncated } = truncateLabel(label);
            const labelWidth = Math.max(44, displayLabel.length * 7.6 + 24);
            const labelHeight = 22;
            const labelX = 22;
            const labelY = -labelHeight / 2;

            return (
              <g>
                {truncated && <title>{label}</title>}
                <circle r={14} fill={statusColor(node.status)} />
                <rect
                  x={labelX}
                  y={labelY}
                  width={labelWidth}
                  height={labelHeight}
                  rx={10}
                  fill="rgba(2, 6, 23, 0.95)"
                  stroke="#475569"
                />
                <text
                  x={labelX + labelWidth / 2}
                  y={0}
                  fill="#f8fafc"
                  fontSize={LABEL_FONT_SIZE}
                  fontWeight={LABEL_FONT_WEIGHT}
                  stroke={LABEL_STROKE}
                  strokeWidth={LABEL_STROKE_WIDTH}
                  strokeLinejoin="round"
                  paintOrder="stroke"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {displayLabel}
                </text>
              </g>
            );
          }}
        />
      )}

      <div className="legend">
        <div className="legend-item">
          <span className="legend-dot accepted" />
          <span>Accepted</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot rejected" />
          <span>Rejected</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot pending" />
          <span>Pending</span>
        </div>
      </div>

      <div className="hud">
        <div>TPS: {tokensPerSecond.toFixed(2)}</div>
        <div>Speedup: {speedup ? speedup.toFixed(2) : "—"}</div>
      </div>
    </div>
  );
}