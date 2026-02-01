import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import TreeViz from "./components/TreeViz";
import type { TreeNode } from "./components/TreeViz";

const initialTree: TreeNode = { name: "root", children: [] };

type TreeDebug = {
  candidates: string[];
  accepted: boolean[];
  speedup?: number;
};

type StreamPayload = {
  text: string;
  tree_debug: TreeDebug;
  done?: boolean;
};

export default function App() {
  const [tree, setTree] = useState<TreeNode>(initialTree);
  const [output, setOutput] = useState("");
  const [tokens, setTokens] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [speedup, setSpeedup] = useState<number | null>(null);
  const [status, setStatus] = useState("connecting");
  const [sessionId, setSessionId] = useState(0);
  const retryRef = useRef(0);
  const doneRef = useRef(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retryTimer: number | null = null;

    doneRef.current = false;

    const connect = () => {
      const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
      const wsUrl = `${wsProtocol}://${window.location.hostname}:8000/generate_stream`;
      setStatus("connecting");
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("open");
      };

      ws.onclose = () => {
        if (doneRef.current) {
          setStatus("done");
          return;
        }
        setStatus("closed");
        const delay = Math.min(5000, 500 * 2 ** retryRef.current);
        retryRef.current += 1;
        retryTimer = window.setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws?.close();
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data) as StreamPayload;

        if (data.done) {
          doneRef.current = true;
          setStatus("done");
          return;
        }

        setOutput((prev) => prev + (data.text ?? ""));
        setTokens((prev) => prev + 1);
        setSpeedup(data.tree_debug?.speedup ?? null);
        setStartTime((prev) => prev ?? performance.now());

        setTree((prev) => {
          const step = (prev.children?.length ?? 0) + 1;
          const candidates = data.tree_debug?.candidates ?? [];
          const accepted = data.tree_debug?.accepted ?? [];

          const children: TreeNode[] = candidates.map((name, i) => ({
            name,
            status: accepted[i] ? "accepted" : "rejected",
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
      if (retryTimer) {
        window.clearTimeout(retryTimer);
      }
      ws?.close();
    };
  }, [sessionId]);

  const tokensPerSecond = useMemo(() => {
    if (!startTime) return 0;
    const elapsed = (performance.now() - startTime) / 1000;
    return elapsed > 0 ? tokens / elapsed : 0;
  }, [tokens, startTime]);

  const handleReplay = () => {
    setTree(initialTree);
    setOutput("");
    setTokens(0);
    setStartTime(null);
    setSpeedup(null);
    setSessionId((prev) => prev + 1);
  };

  return (
    <div className="app">
      <div className="left">
        <h1>Project Gorgon</h1>
        <div className="status-row">
          <div className={`status ${status}`}>WS: {status}</div>
          <button className="replay" onClick={handleReplay}>
            Replay
          </button>
        </div>
        <TreeViz data={tree} tokensPerSecond={tokensPerSecond} speedup={speedup} />
      </div>
      <div className="right">
        <h2>Stream Output</h2>
        <div className="output">{output}</div>
      </div>
    </div>
  );
}