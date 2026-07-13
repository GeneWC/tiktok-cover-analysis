import { describe, expect, it } from "vitest";
import {
  getDefinition,
  getInlineDefinition,
  getSignalTerm,
} from "./glossary";

describe("getDefinition", () => {
  it("looks up by feature key", () => {
    expect(getDefinition("audio_silence_ratio")).toMatch(/silent/i);
    expect(getDefinition("colorfulness_full")).toMatch(/colors/i);
  });
  it("looks up by human label (case/space-insensitive)", () => {
    expect(getDefinition("Dead air")).toMatch(/silent/i);
    expect(getDefinition("  AUDIO   CLIPPING ")).toMatch(/distort/i);
  });
  it("returns null for unknown terms", () => {
    expect(getDefinition("totally unknown metric")).toBeNull();
  });
});

describe("getSignalTerm", () => {
  it("splits the leading term and finds its definition", () => {
    const parsed = getSignalTerm(
      "Dead air: weaker than most comparable covers."
    );
    expect(parsed?.term).toBe("Dead air");
    expect(parsed?.rest).toBe(": weaker than most comparable covers.");
    expect(parsed?.definition).toMatch(/silent/i);
  });
  it("returns null when the term is unknown or has no colon", () => {
    expect(getSignalTerm("Something: with no glossary entry.")).toBeNull();
    expect(getSignalTerm("no colon here")).toBeNull();
  });
});

describe("getInlineDefinition", () => {
  it("finds known jargon inside a recommendation sentence", () => {
    const hit = getInlineDefinition(
      "Preserve dynamic range; avoid over-compressing the audio."
    );
    expect(hit?.phrase).toBe("dynamic range");
    expect(hit?.definition).toMatch(/quietest and loudest/i);
  });
  it("returns null when no jargon is present", () => {
    expect(getInlineDefinition("Center the performer in the frame.")).toBeNull();
  });
});
