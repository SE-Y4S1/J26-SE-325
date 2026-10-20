const crypto = require("crypto");
const jwt = require("jsonwebtoken");

/**
 * Verification of the platform service's tokens, mirroring Component 1's service/auth.py so
 * every service accepts exactly the same tokens:
 *   HS256 only (a caller-chosen algorithm is how "alg: none" and RS256->HS256 attacks work),
 *   issuer "j26-se-325-platform", and exp, iat and sub all required.
 *
 * AUTH_REQUIRED gates enforcement and defaults to off, as in Component 1, so existing callers
 * keep working until a deployment turns it on. A valid token is still read when enforcement
 * is off, so separation of duties applies whenever the caller is identifiable.
 *
 * Tokens carry no role claim. Rights therefore come from this service's own configuration:
 *   AUDIT_REVIEWER_IDS  comma-separated platform user ids allowed to review and audit;
 *   sub "service:platform" (signed with the shared secret) is the platform service itself,
 *                       allowed to anchor decisions and to record step-up verifications.
 */

const ALGORITHM = "HS256";
const ISSUER = "j26-se-325-platform";
const PLATFORM_SERVICE_SUBJECT = "service:platform";

class AuthError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function authRequired() {
  return ["1", "true", "yes", "on"].includes(String(process.env.AUTH_REQUIRED || "").trim().toLowerCase());
}

function jwtSecret() {
  return String(process.env.JWT_SECRET || "").trim();
}

function reviewerIds() {
  return new Set(String(process.env.AUDIT_REVIEWER_IDS || "").split(",").map((s) => s.trim()).filter(Boolean));
}

/**
 * Keyed pseudonymous reference to a platform user: HMAC-SHA256(K, "c3-subject:" + id).
 *
 * An unkeyed hash of a small integer id is reversible by enumeration; a keyed one is not
 * without K. K is derived from JWT_SECRET with HKDF (label "c3-subject-ref"), so the platform
 * and this service compute the same reference with no new shared secret; AUDIT_PSEUDONYM_KEY
 * overrides it when the two must differ. Returns null when no key is configured.
 */
function pseudonymKey() {
  const override = String(process.env.AUDIT_PSEUDONYM_KEY || "").trim();
  if (override) return Buffer.from(override, "utf8");
  const secret = jwtSecret();
  if (!secret) return null;
  return Buffer.from(crypto.hkdfSync("sha256", Buffer.from(secret, "utf8"), Buffer.alloc(0), Buffer.from("c3-subject-ref", "utf8"), 32));
}

function subjectRef(userId) {
  const key = pseudonymKey();
  if (!key) return null;
  return crypto.createHmac("sha256", key).update(`c3-subject:${userId}`).digest("hex");
}

function verifyToken(token) {
  const secret = jwtSecret();
  if (!secret) {
    throw new AuthError(500, "Server auth is misconfigured");
  }
  let payload;
  try {
    payload = jwt.verify(token, secret, { algorithms: [ALGORITHM], issuer: ISSUER });
  } catch (err) {
    throw new AuthError(401, err.name === "TokenExpiredError" ? "Token expired" : "Not authenticated");
  }
  for (const claim of ["exp", "iat", "sub"]) {
    if (payload[claim] === undefined || payload[claim] === null || payload[claim] === "") {
      throw new AuthError(401, "Not authenticated");
    }
  }
  return payload;
}

function callerFrom(payload) {
  const sub = String(payload.sub);
  const isService = sub === PLATFORM_SERVICE_SUBJECT;
  return {
    sub,
    isService,
    isReviewer: !isService && reviewerIds().has(sub),
    ref: isService ? PLATFORM_SERVICE_SUBJECT : subjectRef(sub)
  };
}

/**
 * Express middleware: attaches req.caller, or null when anonymous.
 *
 * With AUTH_REQUIRED on, an invalid token is refused (401, or 500 when no secret is set).
 * With it off, an invalid or unverifiable token is treated as anonymous rather than refused:
 * this service ignored Authorization headers entirely before, and a client sending an expired
 * session token must not start failing (or be signed out) because of it.
 */
function identify(req, res, next) {
  const header = req.get("authorization") || "";
  const match = /^Bearer\s+(.+)$/i.exec(header);
  req.caller = null;
  if (!match) {
    return next();
  }
  try {
    req.caller = callerFrom(verifyToken(match[1].trim()));
    return next();
  } catch (err) {
    if (!authRequired()) return next();
    return res.status(err.status).set("WWW-Authenticate", "Bearer").json({ success: false, message: err.message });
  }
}

/**
 * Guards a route. `allow` lists who may call it when AUTH_REQUIRED is on:
 *   "reviewer" -- a user in AUDIT_REVIEWER_IDS;  "service" -- the platform service.
 * With enforcement off the route stays open, exactly as before.
 */
function requireRole(...allow) {
  return (req, res, next) => {
    if (!authRequired()) return next();
    if (!jwtSecret()) {
      return res.status(500).json({ success: false, message: "Server auth is misconfigured" });
    }
    if (!req.caller) {
      return res.status(401).set("WWW-Authenticate", "Bearer").json({ success: false, message: "Not authenticated" });
    }
    const permitted = (allow.includes("reviewer") && req.caller.isReviewer) || (allow.includes("service") && req.caller.isService);
    if (!permitted) {
      return res.status(403).json({ success: false, message: "Not permitted to perform this audit action" });
    }
    return next();
  };
}

/** Whether this caller may see full record contents (amounts, scores, canonical payloads). */
function canSeeRecordContents(req) {
  if (!authRequired()) return true;
  return Boolean(req.caller && (req.caller.isReviewer || req.caller.isService));
}

module.exports = {
  ALGORITHM,
  ISSUER,
  PLATFORM_SERVICE_SUBJECT,
  authRequired,
  subjectRef,
  verifyToken,
  identify,
  requireRole,
  canSeeRecordContents,
  AuthError
};
