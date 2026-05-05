import federation from "@originjs/vite-plugin-federation";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
//import legacy from "@vitejs/plugin-legacy";
/**TODO legacy addon*/

export default defineConfig({
  plugins: [
    react(),
    federation({
      name: "arm_app",
      filename: "remoteEntry.js",
      exposes: {
        "./ARMApp": "./src/App.jsx",
      },
      shared: ["react", "react-dom"],
    }),
  ],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    target: "esnext",
  },
});
