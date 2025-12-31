
/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "#ffffff",
                foreground: "#000000",
                accent: "#ff00ff", // Example vibrant accent
                "brutal-shadow": "#000000",
            },
            boxShadow: {
                "neo": "8px 8px 0px 0px #000000",
                "neo-hover": "4px 4px 0px 0px #000000",
                "neo-active": "0px 0px 0px 0px #000000",
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                mono: ['"JetBrains Mono"', 'monospace'],
                display: ['Oswald', 'sans-serif'],
            },
        },
    },
    plugins: [],
}
