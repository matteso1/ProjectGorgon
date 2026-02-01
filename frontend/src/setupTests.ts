import "@testing-library/jest-dom/vitest";

class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (!("ResizeObserver" in globalThis)) {
  // @ts-expect-error - runtime stub for tests
  globalThis.ResizeObserver = ResizeObserver;
}

Object.defineProperty(HTMLElement.prototype, "clientWidth", {
  configurable: true,
  get: () => 800,
});

Object.defineProperty(HTMLElement.prototype, "clientHeight", {
  configurable: true,
  get: () => 600,
});
