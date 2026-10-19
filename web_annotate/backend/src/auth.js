// Minimal signed-token session. Raters log in with a shared access code and
// a display name — no passwords, no email, nothing personally identifying
// beyond the name they choose. The token only carries the rater id.
import crypto from "node:crypto";

const SECRET = process.env.SESSION_SECRET;
if (!SECRET || SECRET === "change-me-to-a-long-random-string" || SECRET.length < 32) {
  throw new Error("Set SESSION_SECRET to a random string of at least 32 characters");
}
const TTL_DAYS = 60;

export function sign(raterId) {
  const exp = Date.now() + TTL_DAYS * 86_400_000;
  const body = `${raterId}.${exp}`;
  const sig = crypto.createHmac("sha256", SECRET).update(body).digest("base64url");
  return `${body}.${sig}`;
}

export function verify(token) {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  const [raterId, exp, sig] = parts;
  const expect = crypto
    .createHmac("sha256", SECRET)
    .update(`${raterId}.${exp}`)
    .digest("base64url");
  const a = Buffer.from(sig);
  const b = Buffer.from(expect);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  if (Number(exp) < Date.now()) return null;
  return raterId;
}

// Express middleware: puts req.raterId in place or answers 401.
export function requireRater(req, res, next) {
  const header = req.get("authorization") || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;
  const raterId = verify(token);
  if (!raterId) return res.status(401).json({ error: "session expired" });
  req.raterId = raterId;
  next();
}
