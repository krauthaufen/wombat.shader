// Real-Dawn WGSL validator for tests. Uses the `webgpu` npm package
// (Google's `dawn.node` binding) to spin up a GPU device and feed
// emitted shader source through `device.createShaderModule()` +
// `getCompilationInfo()`. Catches anything Dawn would catch in the
// browser: type mismatches, redeclarations, unresolved identifiers,
// shift-amount-not-u32, struct field-offset / alignment issues,
// builtin-slot violations, etc.
//
// One device is created on first use and reused. `validateWgsl`
// returns `{ ok, errors, warnings }`. Tests use `expectValid` /
// `expectInvalid` for clean assertions.

import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const _navigator = { gpu: create([]) };

let devicePromise: Promise<GPUDevice> | null = null;
function getDevice(): Promise<GPUDevice> {
  if (!devicePromise) {
    devicePromise = (async () => {
      const adapter = await _navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("dawn.node: no GPU adapter — install Dawn deps");
      return await adapter.requestDevice();
    })();
  }
  return devicePromise;
}

export interface ValidationResult {
  readonly ok: boolean;
  readonly errors: readonly { message: string; line: number; col: number }[];
  readonly warnings: readonly { message: string; line: number; col: number }[];
  readonly source: string;
}

export async function validateWgsl(source: string): Promise<ValidationResult> {
  const dev = await getDevice();
  const mod = dev.createShaderModule({ code: source });
  const info = await mod.getCompilationInfo();
  const errors: { message: string; line: number; col: number }[] = [];
  const warnings: { message: string; line: number; col: number }[] = [];
  for (const m of info.messages) {
    const entry = { message: m.message, line: m.lineNum, col: m.linePos };
    if (m.type === "error") errors.push(entry);
    else if (m.type === "warning") warnings.push(entry);
  }
  return { ok: errors.length === 0, errors, warnings, source };
}

/**
 * Format a validation failure with line context — easier to debug
 * than just dumping the raw error.
 */
export function formatFailure(r: ValidationResult): string {
  const lines = r.source.split("\n");
  const out: string[] = ["WGSL validation failed:"];
  for (const e of r.errors) {
    out.push(`  ${e.line}:${e.col}: ${e.message}`);
    const ctxStart = Math.max(1, e.line - 1);
    const ctxEnd = Math.min(lines.length, e.line + 1);
    for (let i = ctxStart; i <= ctxEnd; i++) {
      out.push(`    ${i.toString().padStart(4)} | ${lines[i - 1]}`);
    }
  }
  return out.join("\n");
}

export async function expectValidWgsl(source: string): Promise<void> {
  const r = await validateWgsl(source);
  if (!r.ok) throw new Error(formatFailure(r));
}

/** Asserts the WGSL is INVALID and the error matches a regex. */
export async function expectInvalidWgsl(source: string, errorPattern: RegExp): Promise<void> {
  const r = await validateWgsl(source);
  if (r.ok) throw new Error("expected WGSL to fail validation but it passed:\n" + source);
  const found = r.errors.some((e) => errorPattern.test(e.message));
  if (!found) {
    throw new Error(
      `error didn't match ${errorPattern}\n  errors:\n` +
      r.errors.map((e) => `    ${e.line}:${e.col}: ${e.message}`).join("\n"),
    );
  }
}
