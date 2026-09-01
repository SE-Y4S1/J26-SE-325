/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: "#0b0f19",
          card: "#111827",
          border: "#1f293d",
          accent: "#3b82f6",
          emerald: "#10b981",
          rose: "#f43f5e",
          amber: "#f59e0b",
          purple: "#8b5cf6"
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'Plus Jakarta Sans', 'sans-serif']
      }
    },
  },
  plugins: [],
}
