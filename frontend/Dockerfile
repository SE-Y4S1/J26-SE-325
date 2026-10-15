# Next.js 16 frontend.
#
# NEXT_PUBLIC_* values are inlined at BUILD time, not read at runtime, so the service URLs
# must be supplied as build args. Changing where a backend lives means rebuilding this image
# — that is a property of NEXT_PUBLIC_, not a choice made here.
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

ARG NEXT_PUBLIC_PLATFORM_URL=http://localhost:8100
ARG NEXT_PUBLIC_PORTFOLIO_URL=http://localhost:8000
ARG NEXT_PUBLIC_FRAUD_URL=http://localhost:8001
ARG NEXT_PUBLIC_AUDIT_URL=http://localhost:8002
ARG NEXT_PUBLIC_ASSISTANT_URL=http://localhost:8003
ENV NEXT_PUBLIC_PLATFORM_URL=$NEXT_PUBLIC_PLATFORM_URL \
    NEXT_PUBLIC_PORTFOLIO_URL=$NEXT_PUBLIC_PORTFOLIO_URL \
    NEXT_PUBLIC_FRAUD_URL=$NEXT_PUBLIC_FRAUD_URL \
    NEXT_PUBLIC_AUDIT_URL=$NEXT_PUBLIC_AUDIT_URL \
    NEXT_PUBLIC_ASSISTANT_URL=$NEXT_PUBLIC_ASSISTANT_URL \
    NEXT_TELEMETRY_DISABLED=1

RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1

RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001

# `output: "standalone"` in next.config.ts produces a self-contained server with only the
# modules actually imported, which is far smaller than shipping node_modules.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:3000/ || exit 1

CMD ["node", "server.js"]
