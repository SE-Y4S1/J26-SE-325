import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits a self-contained server bundle with only the modules actually imported, which is
  // what frontend/Dockerfile copies. Without it the runtime stage would need the whole
  // node_modules tree.
  //
  // Not on Vercel (which sets VERCEL=1 at build time): its build adapter packages the output
  // itself and fails looking for .next/next-server.js.nft.json when standalone is on.
  output: process.env.VERCEL ? undefined : "standalone",
};

export default nextConfig;
