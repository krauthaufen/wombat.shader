import { defineConfig } from "vite";
import { boperators } from "@boperators/plugin-vite";

export default defineConfig({
  plugins: [boperators()],
  resolve: {
    preserveSymlinks: false,
  },
  optimizeDeps: {
    // Force pre-bundling of typescript so the runtime's frontend can use it
    // in the browser without ESM resolution oddities.
    include: ["typescript"],
  },
});
