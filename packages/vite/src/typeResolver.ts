// TypeScript type-checker integration.
//
// The plugin owns one `ts.LanguageService` per build, configured from
// the user's `tsconfig.json`. We update its in-memory snapshot for the
// file currently being transformed (so ESM module resolution still
// sees the version on disk for everything else) and query the checker
// to:
//
//   - Resolve the IR type of a closure-captured identifier (replaces
//     the previous "annotated declaration or `new V*f(...)` initialiser"
//     heuristic — types now come from anywhere TS understands them).
//   - Decide whether a free identifier is a uniform binding (its
//     declaration is `declare const ...`, ambient) or a closure
//     capture (a regular `const/let/var` with a runtime value).
//
// This means uniforms can live in a base library:
//
//     // base.d.ts
//     declare const u: { readonly mvp: M44f; readonly tint: V3f };
//
//     // app.ts
//     const fs = fragment(input => ({ outColor: vec4(u.tint, 1.0) }));
//
// The plugin asks the checker for `u`'s symbol, sees it's declared
// ambient in `base.d.ts`, walks its type literal, and emits a
// `Uniform` ValueDef alongside the IR template.

import * as path from "node:path";
import ts from "typescript";

export interface TypeResolverOptions {
  readonly rootDir: string;
  /** Optional explicit tsconfig path; auto-discovered if omitted. */
  readonly tsconfigPath?: string;
}

export class TypeResolver {
  private readonly rootDir: string;
  private readonly host: ts.LanguageServiceHost;
  private readonly service: ts.LanguageService;
  private readonly versions = new Map<string, number>();
  private readonly overrides = new Map<string, string>();
  private readonly rootFileNames: readonly string[];

  constructor(opts: TypeResolverOptions) {
    this.rootDir = opts.rootDir;
    const configPath = opts.tsconfigPath
      ?? ts.findConfigFile(opts.rootDir, ts.sys.fileExists, "tsconfig.json");
    if (!configPath) {
      throw new Error(
        `wombat.shader: no tsconfig.json found at or above ${opts.rootDir}. ` +
        `Provide \`tsconfigPath\` to the plugin if it lives elsewhere.`,
      );
    }
    const cfg = ts.readConfigFile(configPath, ts.sys.readFile);
    if (cfg.error) {
      throw new Error(`wombat.shader: failed to read ${configPath} — ${ts.flattenDiagnosticMessageText(cfg.error.messageText, "\n")}`);
    }
    const parsed = ts.parseJsonConfigFileContent(
      cfg.config, ts.sys, path.dirname(configPath),
    );
    this.rootFileNames = parsed.fileNames;

    this.host = {
      getScriptFileNames: () => {
        const set = new Set<string>(this.rootFileNames);
        for (const f of this.overrides.keys()) set.add(f);
        return [...set];
      },
      getScriptVersion: (fileName) => String(this.versions.get(fileName) ?? 0),
      getScriptSnapshot: (fileName) => {
        const override = this.overrides.get(fileName);
        if (override !== undefined) return ts.ScriptSnapshot.fromString(override);
        if (!ts.sys.fileExists(fileName)) return undefined;
        const contents = ts.sys.readFile(fileName);
        if (contents === undefined) return undefined;
        return ts.ScriptSnapshot.fromString(contents);
      },
      getCurrentDirectory: () => this.rootDir,
      getCompilationSettings: () => parsed.options,
      getDefaultLibFileName: (o) => ts.getDefaultLibFilePath(o),
      fileExists: (f) => this.overrides.has(f) || ts.sys.fileExists(f),
      readFile: (f) => this.overrides.has(f) ? this.overrides.get(f) : ts.sys.readFile(f),
      readDirectory: ts.sys.readDirectory,
      directoryExists: ts.sys.directoryExists,
      getDirectories: ts.sys.getDirectories,
    };

    this.service = ts.createLanguageService(this.host, ts.createDocumentRegistry());
  }

  /**
   * Update the in-memory copy of a file. Bumps its script version so
   * the LanguageService re-parses on the next query. Call this with
   * the source string Vite hands to the `transform` hook before
   * walking the file.
   */
  setFile(fileName: string, content: string): void {
    const prev = this.overrides.get(fileName);
    if (prev === content) return;
    this.overrides.set(fileName, content);
    this.versions.set(fileName, (this.versions.get(fileName) ?? 0) + 1);
  }

  getProgram(): ts.Program | undefined {
    return this.service.getProgram();
  }

  getChecker(): ts.TypeChecker | undefined {
    return this.getProgram()?.getTypeChecker();
  }

  getSourceFile(fileName: string): ts.SourceFile | undefined {
    return this.getProgram()?.getSourceFile(fileName);
  }
}
