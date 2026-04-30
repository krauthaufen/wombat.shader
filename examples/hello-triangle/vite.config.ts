import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    preserveSymlinks: false,
  },
  optimizeDeps: {
    // Force pre-bundling of typescript so the runtime's frontend can use it
    // in the browser without ESM resolution oddities.
    include: ["typescript"],
  },
});
