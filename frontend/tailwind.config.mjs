/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        activity: {
          1: '#22c55e',  // Standing - verde
          2: '#3b82f6',  // Sitting - azul
          3: '#8b5cf6',  // Lying - morado
          4: '#f59e0b',  // Walking - amarillo
          5: '#ef4444',  // Climbing - rojo
          6: '#06b6d4',  // Waist bends - cyan
          7: '#ec4899',  // Arms elevation - rosa
          8: '#f97316',  // Knees bending - naranja
          9: '#84cc16',  // Cycling - lima
          10: '#14b8a6', // Jogging - teal
          11: '#e11d48', // Running - rosa oscuro
          12: '#6366f1', // Jumping - índigo
        }
      }
    },
  },
  plugins: [],
}
