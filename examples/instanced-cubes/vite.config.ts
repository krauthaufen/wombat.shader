import { defineConfig } from "vite";
export default defineConfig({
  resolve: { preserveSymlinks: false },
  optimizeDeps: { include: ["typescript"] },
});
