// WebGPU — `createShaderModule` for a compiled effect, helper to
// build a render-pipeline descriptor stub from the binding map.

import type { CompiledEffect, CompiledStage } from "../compile.js";

export interface WebGpuShaderModuleSet {
  readonly device: GPUDevice;
  readonly vertex?: GPUShaderModule;
  readonly fragment?: GPUShaderModule;
  readonly compute?: GPUShaderModule;
  readonly bindings: BindingPlan;
  dispose(): void;
}

export interface BindingPlan {
  /** group → list of (binding-slot, kind, info) */
  readonly groups: ReadonlyMap<number, readonly BindingEntry[]>;
}

export interface BindingEntry {
  readonly slot: number;
  readonly kind: "uniform" | "sampler" | "texture" | "storage";
  readonly name: string;
}

export function createShaderModules(
  device: GPUDevice,
  effect: CompiledEffect,
): WebGpuShaderModuleSet {
  if (effect.target !== "wgsl") {
    throw new Error(`createShaderModules (WebGPU) requires a WGSL effect; got ${effect.target}`);
  }
  const find = (s: "vertex" | "fragment" | "compute"): CompiledStage | undefined =>
    effect.stages.find((x) => x.stage === s);
  const make = (stage: CompiledStage | undefined): GPUShaderModule | undefined =>
    stage ? device.createShaderModule({ code: stage.source }) : undefined;

  const vertex = make(find("vertex"));
  const fragment = make(find("fragment"));
  const compute = make(find("compute"));

  const bindings = collectBindings(effect);

  return {
    device,
    ...(vertex !== undefined ? { vertex } : {}),
    ...(fragment !== undefined ? { fragment } : {}),
    ...(compute !== undefined ? { compute } : {}),
    bindings,
    dispose() {
      // GPUShaderModule has no explicit destroy; GC handles it.
    },
  };
}

function collectBindings(effect: CompiledEffect): BindingPlan {
  const groups = new Map<number, BindingEntry[]>();
  const push = (group: number, entry: BindingEntry): void => {
    let arr = groups.get(group);
    if (!arr) { arr = []; groups.set(group, arr); }
    if (!arr.some((e) => e.slot === entry.slot && e.name === entry.name)) {
      arr.push(entry);
    }
  };
  for (const stage of effect.stages) {
    const b = stage.bindings as unknown as {
      uniforms?: readonly { name: string; group: number; slot: number }[];
      samplers?: readonly { name: string; group: number; slot: number }[];
      storage?: readonly { name: string; group: number; slot: number }[];
    };
    for (const u of b.uniforms ?? []) push(u.group, { slot: u.slot, kind: "uniform", name: u.name });
    for (const s of b.samplers ?? []) push(s.group, { slot: s.slot, kind: "sampler", name: s.name });
    for (const s of b.storage ?? []) push(s.group, { slot: s.slot, kind: "storage", name: s.name });
  }
  // Sort each group by slot for deterministic output.
  for (const arr of groups.values()) arr.sort((a, b) => a.slot - b.slot);
  return { groups };
}
