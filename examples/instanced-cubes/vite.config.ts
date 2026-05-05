import { defineConfig } from "vite";
import { boperators } from "@boperators/plugin-vite";
export default defineConfig({
  plugins: [boperators()],
  resolve: { preserveSymlinks: false },
  optimizeDeps: { include: ["typescript"] },
});
