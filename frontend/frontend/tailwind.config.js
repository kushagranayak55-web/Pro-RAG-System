/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: "#080B12",
        surface: "#0E121C",
        elevated: "#141926",
        raised: "#1A2033",
        border: {
          DEFAULT: "#212739",
          soft: "#191F2E",
        },
        ink: {
          DEFAULT: "#E8EAF0",
          muted: "#8991A6",
          faint: "#5B6478",
        },
        accent: {
          blue: "#5B8DEF",
          purple: "#9D7BEF",
          green: "#34D399",
          amber: "#F0B357",
          red: "#F0645B",
        },
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'Inter'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 0 rgba(255,255,255,0.02) inset, 0 8px 24px -12px rgba(0,0,0,0.6)",
        glow: "0 0 0 1px rgba(91,141,239,0.25), 0 0 24px -4px rgba(91,141,239,0.35)",
      },
      keyframes: {
        pulseDot: {
          "0%, 100%": { opacity: 0.4, transform: "scale(1)" },
          "50%": { opacity: 1, transform: "scale(1.15)" },
        },
        flow: {
          "0%": { strokeDashoffset: "24" },
          "100%": { strokeDashoffset: "0" },
        },
        riseIn: {
          "0%": { opacity: 0, transform: "translateY(6px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
      },
      animation: {
        pulseDot: "pulseDot 1.4s ease-in-out infinite",
        flow: "flow 1s linear infinite",
        riseIn: "riseIn 0.35s ease-out",
      },
    },
  },
  plugins: [],
};
