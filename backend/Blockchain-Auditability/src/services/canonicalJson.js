/**
 * Deterministic JSON serialisation for anything that is hashed.
 *
 * Object keys are sorted at every depth, arrays keep their order, and values that JSON cannot
 * represent faithfully (NaN, Infinity, undefined, functions) are rejected rather than silently
 * turned into null or dropped -- a hash over a value that changed shape on the way in proves
 * nothing.
 *
 * This replaces JSON.stringify(obj, sortedTopLevelKeys), which uses the key array as an
 * allow-list at EVERY depth and so serialised every nested object as {} (see
 * computePolicyHashV1 in policyDefinition.js).
 */
function canonicalJson(value) {
  return serialise(value, "$");
}

function serialise(value, path) {
  if (value === null) return "null";
  switch (typeof value) {
    case "string":
    case "boolean":
      return JSON.stringify(value);
    case "number":
      if (!Number.isFinite(value)) {
        throw new TypeError(`canonicalJson: non-finite number at ${path}`);
      }
      return JSON.stringify(value);
    case "object":
      if (Array.isArray(value)) {
        return `[${value.map((item, i) => serialise(item, `${path}[${i}]`)).join(",")}]`;
      }
      return `{${Object.keys(value)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${serialise(value[key], `${path}.${key}`)}`)
        .join(",")}}`;
    default:
      throw new TypeError(`canonicalJson: unsupported ${typeof value} at ${path}`);
  }
}

module.exports = { canonicalJson };
