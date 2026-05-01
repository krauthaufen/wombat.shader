// v3 source-map utilities. Both WGSL and GLSL emitters track which
// `Span` produced each emitted line, then call `buildSourceMap` to
// produce the standard JSON shape attached to `CompiledStage`.
//
// We emit *line-granular* maps: every emitted line maps to a single
// (file, line, col) of the originating TS source. That's enough for
// "GPU compiler rejected line N → click here" navigation in the
// browser dev-tools. Per-token granularity would require threading
// per-Expr spans into the Writer; not worth the cost yet.
//
// The base64-VLQ encoding follows the v3 source-map spec verbatim.

import type { Span } from "./types.js";

export interface SourceMap {
  readonly version: 3;
  readonly sources: ReadonlyArray<string>;
  readonly sourcesContent: ReadonlyArray<string | null>;
  readonly mappings: string;
  readonly file?: string;
}

const VLQ_CHARS =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function vlqEncodeSingle(value: number): string {
  let v = value < 0 ? ((-value) << 1) | 1 : value << 1;
  let out = "";
  do {
    let digit = v & 0b11111;
    v >>>= 5;
    if (v > 0) digit |= 0b100000;
    out += VLQ_CHARS[digit];
  } while (v > 0);
  return out;
}

function vlqEncode(values: ReadonlyArray<number>): string {
  let out = "";
  for (const v of values) out += vlqEncodeSingle(v);
  return out;
}

/**
 * Convert a 0-indexed character offset to a 0-indexed (line, col)
 * position. Cheap to call repeatedly for one source string;
 * memoise the line table outside if you need it for many spans.
 */
function positionOfOffset(source: string, offset: number): { line: number; col: number } {
  let line = 0;
  let lineStart = 0;
  for (let i = 0; i < offset && i < source.length; i++) {
    if (source.charCodeAt(i) === 10 /* '\n' */) {
      line++;
      lineStart = i + 1;
    }
  }
  return { line, col: offset - lineStart };
}

export interface LineSegment {
  /** 0-based generated column. */
  readonly col: number;
  readonly span: Span;
}

export interface BuildSourceMapInput {
  /**
   * One entry per emitted line. Each entry is the (possibly empty)
   * list of mapping segments for that line, sorted by ascending
   * `col`. Empty array = unmapped line.
   *
   * Legacy callers may still pass `lineSpans` (line-granular maps);
   * the runtime converts them on the fly.
   */
  readonly lineSegments?: ReadonlyArray<readonly LineSegment[]>;
  /** Legacy: one span per line. Mapped to a single col-0 segment per line. */
  readonly lineSpans?: ReadonlyArray<Span | undefined>;
  /** File contents lookup keyed on `Span.file`. */
  readonly fileContents: ReadonlyMap<string, string>;
  /** Optional output filename written into the map's `file` field. */
  readonly outputFile?: string;
}

export function buildSourceMap(input: BuildSourceMapInput): SourceMap {
  const sources: string[] = [];
  const sourcesContent: (string | null)[] = [];
  const fileIndex = new Map<string, number>();

  function indexOfFile(file: string): number {
    let idx = fileIndex.get(file);
    if (idx === undefined) {
      idx = sources.length;
      fileIndex.set(file, idx);
      sources.push(file);
      sourcesContent.push(input.fileContents.get(file) ?? null);
    }
    return idx;
  }

  // Per-source absolute line/col tables — built lazily for any source
  // that has at least one mapped span.
  const positionCache = new Map<string, (offset: number) => { line: number; col: number }>();
  function pos(file: string, offset: number): { line: number; col: number } {
    let fn = positionCache.get(file);
    if (!fn) {
      const contents = input.fileContents.get(file) ?? "";
      fn = (o) => positionOfOffset(contents, o);
      positionCache.set(file, fn);
    }
    return fn(offset);
  }

  // Resolve to a uniform `lineSegments` view.
  const lineSegments: readonly (readonly LineSegment[])[] =
    input.lineSegments
      ?? (input.lineSpans ?? []).map(s => s ? [{ col: 0, span: s }] : []);

  // VLQ deltas are relative to the previous segment (across lines).
  // Generated-column delta resets at the start of each line per the
  // v3 spec; we track `prevGenCol` and reset it on `;` boundaries.
  let prevSourceIndex = 0;
  let prevSourceLine = 0;
  let prevSourceCol = 0;

  const lines: string[] = [];
  for (const segments of lineSegments) {
    if (segments.length === 0) {
      lines.push("");
      continue;
    }
    let lineEncoded = "";
    let prevGenCol = 0;
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i]!;
      const idx = indexOfFile(seg.span.file);
      const { line, col } = pos(seg.span.file, seg.span.start);
      const fields = [
        seg.col - prevGenCol,
        idx - prevSourceIndex,
        line - prevSourceLine,
        col - prevSourceCol,
      ];
      if (i > 0) lineEncoded += ",";
      lineEncoded += vlqEncode(fields);
      prevGenCol = seg.col;
      prevSourceIndex = idx;
      prevSourceLine = line;
      prevSourceCol = col;
    }
    lines.push(lineEncoded);
  }

  const out: SourceMap = {
    version: 3,
    sources,
    sourcesContent,
    mappings: lines.join(";"),
    ...(input.outputFile !== undefined ? { file: input.outputFile } : {}),
  };
  return out;
}
