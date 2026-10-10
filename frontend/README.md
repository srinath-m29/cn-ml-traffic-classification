# Frontend

React and TypeScript dashboard built with Vite.

From this directory:

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```

Set `VITE_API_BASE_URL` in `.env.local` to the FastAPI base URL when it is not `http://127.0.0.1:8000`. The matching backend origin must be allowed by its `CORS_ORIGINS` setting.

Run `npm run build` to create the production bundle in `dist/`.
