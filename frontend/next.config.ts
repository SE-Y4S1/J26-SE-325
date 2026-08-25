import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits a self-contained server bundle with only the modules actually imported, which is
  // what frontend/Dockerfile copies. Without it the runtime stage would need the whole
  // node_modules tree.
  output: "standalone",
};

export default nextConfig;
