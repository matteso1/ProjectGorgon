import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import TreeViz from "./components/TreeViz";
import type { TreeNode } from "./components/TreeViz";

const initialTree: TreeNode = { name: "root", children: [] };
const DEFAULT_PROMPT = "The future of artificial intelligence is";

type TreeDebug = {
  candidates: string[];
  accepted: boolean[];
  speedup?: number;
  tokens_per_second?: number;
  acceptance_rate?: number;
  total_tokens?: number;
  total_drafted?: number;
  total_accepted?: number;
  iterations?: number;
};

type StreamPayload = {
  text: string;
  tree_debug: TreeDebug;
  done?: boolean;
  error?: string;
};

export default function App() {
  const [tree, setTree] = useState<TreeNode>(initialTree);
  const [output, setOutput] = useState("");
  const [tokens, setTokens] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [speedup, setSpeedup] = useState<number | null>(null);
  const [acceptanceRate, setAcceptanceRate] = useState<number | null>(null);
  const [status, setStatus] = useState("idle");
  const [sessionId, setSessionId] = useState(0);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [drafted, setDrafted] = useState(0);
  const [accepted, setAccepted] = useState(0);
  const [iterations, setIterations] = useState(0);
  const retryRef = useRef(0);
  const doneRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);

  const handleGenerate = () => {
    setTree(initialTree);
    setOutput("");
    setTokens(0);
    setStartTime(null);
    setSpeedup(null);
    setAcceptanceRate(null);
    setDrafted(0);
    setAccepted(0);
    setIterations(0);
    doneRef.current = false;
    setSessionId((prev) => prev + 1);
  };

  useEffect(() => {
    if (sessionId === 0) return;

    let ws: WebSocket | null = null;
    let retryTimer: number | null = null;
    doneRef.current = false;

    const connect = () => {
      const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
      const wsUrl = `${wsProtocol}://${window.location.hostname}:8000/generate_stream`;
      setStatus("connecting");
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("streaming");
        ws?.send(JSON.stringify({
          prompt: prompt,
          max_new_tokens: 128,
        }));
      };

      ws.onclose = () => {
        if (doneRef.current) {
          setStatus("done");
          return;
        }
        setStatus("reconnecting");
        const delay = Math.min(5000, 500 * 2 ** retryRef.current);
        retryRef.current += 1;
        retryTimer = window.setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws?.close();
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data) as StreamPayload;

        if (data.error) {
          setStatus("error");
          setOutput((prev) => prev + `\n[Error: ${data.error}]`);
          return;
        }

        if (data.done) {
          doneRef.current = true;
          setStatus("done");
          if (data.tree_debug?.total_drafted) setDrafted(data.tree_debug.total_drafted);
          if (data.tree_debug?.total_accepted) setAccepted(data.tree_debug.total_accepted);
          if (data.tree_debug?.iterations) setIterations(data.tree_debug.iterations);
          return;
        }

        setOutput((prev) => prev + (data.text ?? ""));
        setTokens((prev) => prev + 1);
        setSpeedup(data.tree_debug?.speedup ?? null);
        setAcceptanceRate(data.tree_debug?.acceptance_rate ?? null);
        setStartTime((prev) => prev ?? performance.now());

        setTree((prev) => {
          const step = (prev.children?.length ?? 0) + 1;
          const candidates = data.tree_debug?.candidates ?? [];
          const acceptedFlags = data.tree_debug?.accepted ?? [];

          const children: TreeNode[] = candidates.map((name, i) => ({
            name,
            status: acceptedFlags[i] ? "accepted" : "rejected",
          }));

          const stepNode: TreeNode = {
            name: `step ${step}`,
            status: "pending",
            children,
          };

          return {
            ...prev,
            children: [...(prev.children ?? []), stepNode],
          };
        });
      };
    };

    connect();

    return () => {
      if (retryTimer) window.clearTimeout(retryTimer);
      ws?.close();
    };
  }, [sessionId]);

  const tokensPerSecond = useMemo(() => {
    if (!startTime) return 0;
    const elapsed = (performance.now() - startTime) / 1000;
    return elapsed > 0 ? tokens / elapsed : 0;
  }, [tokens, startTime]);

  return (
    <div className="app">
      <div className="left">
        <div className="header">
          <h1>Project Gorgon<span>speculative decoding</span></h1>
          <div className="controls">
            <div className={`status-badge ${status}`}>{status}</div>
          </div>
        </div>

        <div className="prompt-row">
          <input
            className="prompt-input"
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
            placeholder="Enter a prompt..."
          />
          <button className="btn btn-primary" onClick={handleGenerate}>
            Generate
          </button>
          <button className="btn" onClick={handleGenerate}>
            Replay
          </button>
        </div>

        <TreeViz data={tree} tokensPerSecond={tokensPerSecond} speedup={speedup} />
      </div>

      <div className="right">
        <h2>Live Metrics</h2>
        <div className="metrics-bar">
          <div className="metric-card">
            <div className="metric-label">Tokens/sec</div>
            <div className="metric-value cyan">{tokensPerSecond.toFixed(1)}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Acceptance</div>
            <div className="metric-value green">
              {acceptanceRate !== null ? `${(acceptanceRate * 100).toFixed(0)}%` : "—"}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Speedup</div>
            <div className="metric-value accent">
              {speedup !== null ? `${speedup.toFixed(2)}×` : "—"}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Tokens</div>
            <div className="metric-value">{tokens}</div>
          </div>
        </div>

        <h2>Stream Output</h2>
        <div className="output-panel">
          <div className="output">{output || "Enter a prompt and click Generate..."}</div>
        </div>

        {status === "done" && (
          <>
            <h2>Session Summary</h2>
            <div className="metrics-bar">
              <div className="metric-card">
                <div className="metric-label">Drafted</div>
                <div className="metric-value">{drafted}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Accepted</div>
                <div className="metric-value green">{accepted}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Iterations</div>
                <div className="metric-value">{iterations}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Total</div>
                <div className="metric-value accent">{tokens}</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}