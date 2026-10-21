require("dotenv").config();
const express = require("express");
const cors = require("cors");
const auditRoutes = require("./routes/auditRoutes");
const { identify, authRequired } = require("./auth/auth");

const app = express();
const PORT = process.env.PORT || 8002;

/**
 * Browser origins allowed to call this service, from ALLOWED_ORIGINS like every other service
 * in the platform. Unset keeps the previous behaviour (any origin) so an existing deployment
 * that never set it keeps working -- with a warning, because a production deployment should
 * name its frontend explicitly.
 */
function corsOrigin() {
  const origins = String(process.env.ALLOWED_ORIGINS || "").split(",").map((o) => o.trim()).filter(Boolean);
  if (origins.includes("*")) {
    throw new Error("ALLOWED_ORIGINS must list explicit origins, not '*'.");
  }
  return origins.length ? origins : "*";
}
const origin = corsOrigin();

// Middleware
app.use(cors({ origin, allowedHeaders: ["Content-Type", "Authorization"] }));
app.use(express.json({ limit: "100kb" }));
app.use(identify);

// Routes
app.use("/api/audit", auditRoutes);

// Health Check Endpoint
app.get("/api/health", (req, res) => {
  res.json({
    status: "online",
    component: "Component 03: AI Decision Assurance Engine and tamper-evident audit ledger",
    timestamp: new Date().toISOString()
  });
});

// Unknown routes answer in the service's JSON shape, like every other error.
app.use((req, res) => {
  res.status(404).json({ success: false, message: `No route ${req.method} ${req.path}.` });
});

// Malformed or oversized bodies, and anything a route lets escape, are answered as JSON
// rather than Express's default page, which carries a stack trace outside production.
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  if (err.type === "entity.parse.failed") {
    return res.status(400).json({ success: false, message: "Request body is not valid JSON." });
  }
  if (err.type === "entity.too.large") {
    return res.status(413).json({ success: false, message: "Request body is larger than the 100kb limit." });
  }
  if (err.status >= 400 && err.status < 500 && err.expose) {
    return res.status(err.status).json({ success: false, message: err.message });
  }
  console.error("Unhandled error:", err);
  return res.status(500).json({ success: false, message: "Internal server error." });
});

// Start Server when run directly
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`=============================================================`);
    console.log(` Component 03: AI Decision Assurance Engine + audit ledger`);
    console.log(` Server listening on http://localhost:${PORT}`);
    console.log(` API Endpoint: http://localhost:${PORT}/api/audit`);
    console.log(` Auth: ${authRequired() ? "enforced (AUTH_REQUIRED)" : "not enforced (set AUTH_REQUIRED=true in production)"}`);
    if (origin === "*") {
      console.log(" CORS: any origin (ALLOWED_ORIGINS unset) -- set it to the frontend origin in production");
    }
    console.log(`=============================================================`);
  });
}

module.exports = app;
