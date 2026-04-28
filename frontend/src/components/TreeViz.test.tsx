import { render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import TreeViz, { TreeNode } from "./TreeViz";

vi.mock("react-d3-tree", () => ({
  default: ({ data, renderCustomNodeElement }: any) => (
    <svg data-testid="tree">
      {renderCustomNodeElement({ nodeDatum: data })}
    </svg>
  ),
}));

const baseData: TreeNode = {
  name: "root",
  status: "pending",
};

test("renders legend labels", async () => {
  render(<TreeViz data={baseData} tokensPerSecond={0} speedup={null} />);

  expect(await screen.findByText("Accepted")).toBeInTheDocument();
  expect(screen.getByText("Rejected")).toBeInTheDocument();
  expect(screen.getByText("Pending")).toBeInTheDocument();
});

test("truncates long labels and adds title", async () => {
  const longLabel = "superlongtoken";
  const { container } = render(
    <TreeViz
      data={{ ...baseData, name: longLabel }}
      tokensPerSecond={0}
      speedup={null}
    />
  );

  expect(await screen.findByText("superlong...")).toBeInTheDocument();

  const label = screen.getByText("superlong...");
  expect(label).toHaveAttribute("stroke", "#0b1220");
  expect(label).toHaveAttribute("stroke-width", "3");
  expect(label).toHaveAttribute("paint-order", "stroke");
  expect(label).toHaveAttribute("font-size", "13");

  const title = container.querySelector("title");
  expect(title).not.toBeNull();
  expect(title).toHaveTextContent(longLabel);
});
