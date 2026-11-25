# Web (Next.js + React + TypeScript + Tailwind)

Phase 5 single-page flow, all user-facing text in Bahasa Indonesia: upload a photo (optional storage consent) →
quality checks and predicted face shape with uncalibrated model scores and limitations → hair profile → top 5
recommendations with explanations, tips and score breakdown → delete.

`/api/*` and `/docs` are proxied to the backend (`BACKEND_URL`, default `http://127.0.0.1:8000`), so the browser talks to one origin.

```bash
npm ci
npm run build && npm run start          # http://localhost:3000, backend must be running
npm run lint && npx tsc --noEmit
npx playwright install chromium && npm run test:e2e   # browser test against the running app
```

`SCREENSHOT_DIR=/some/dir npm run test:e2e` also saves screenshots of each step.
