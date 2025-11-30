/** @type {import('tailwindcss').Config} */
module.exports = {
  // CRITICAL: This scans all your component files for Tailwind classes.
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", 
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}