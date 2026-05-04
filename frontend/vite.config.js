import federation from "@originjs/vite-plugin-federation";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
//import legacy from "@vitejs/plugin-legacy";
/**TODO legacy addon*/
import FaviconManifest from "vite-favicon-manifest";

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
    FaviconManifest({
      icon: "public/favicon.ico",
      manifest: {
        name: "Student Risk Platform",
        description: "АРМ исследователя",
      },
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
