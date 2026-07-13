import { afterEach, describe, expect, it, vi } from "vitest";
import { analyze, ApiError, getReport, getStatus } from "./client";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("getStatus", () => {
  it("returns the parsed status body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ analysis_id: "a1", status: "processing", steps: {} })
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await getStatus("a1");
    expect(result.status).toBe("processing");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/analyze/a1/status"),
      undefined
    );
  });
});

describe("getReport", () => {
  it("surfaces the FastAPI detail message on error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: "Analysis 'x' not found." }, { status: 404 })
      )
    );

    await expect(getReport("x")).rejects.toMatchObject({
      status: 404,
      message: "Analysis 'x' not found.",
    });
  });

  it("wraps network failures in an ApiError with status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));
    await expect(getReport("x")).rejects.toBeInstanceOf(ApiError);
    await expect(getReport("x")).rejects.toMatchObject({ status: 0 });
  });
});

describe("analyze", () => {
  it("posts multipart form data and returns the analysis id", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ analysis_id: "a2", status: "processing" })
    );
    vi.stubGlobal("fetch", fetchMock);

    const file = new File([new Uint8Array([1, 2, 3])], "cover.mp4", {
      type: "video/mp4",
    });
    const result = await analyze({ file, instrument: "violin" });

    expect(result.analysis_id).toBe("a2");
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("instrument")).toBe("violin");
  });
});
