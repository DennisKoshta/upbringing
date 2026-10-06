import { initHero } from "./hero.js";
import { initAct1 } from "./act1.js";
import { initAct2 } from "./act2.js";
import { initAct3 } from "./act3.js";
import { initMethods } from "./methods.js";
import { initGlossary } from "./glossary.js";

// ---------------------------------------------------------------- theme
const root = document.documentElement;
try {
  const saved = localStorage.getItem("upbringing-theme");
  if (saved) root.dataset.theme = saved;
} catch { /* storage unavailable: follow the OS */ }
document.getElementById("theme-toggle").addEventListener("click", () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("upbringing-theme", root.dataset.theme); } catch { /* ignore */ }
});

// ---------------------------------------------------------------- chapter highlight
const links = [...document.querySelectorAll(".chapters a")];
const io = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (!e.isIntersecting) return;
    links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
  });
}, { rootMargin: "-40% 0px -55% 0px" });
document.querySelectorAll("section.act").forEach((s) => io.observe(s));

// ---------------------------------------------------------------- lazy init per section
const inits = { act1: initAct1, act2: initAct2, act3: initAct3, methods: initMethods };
const lazy = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (!e.isIntersecting) return;
    lazy.unobserve(e.target);
    inits[e.target.id]().catch((err) => console.error(e.target.id, err));
  });
}, { rootMargin: "800px 0px" });
Object.keys(inits).forEach((id) => lazy.observe(document.getElementById(id)));
initHero().catch((err) => console.error("hero", err));
initGlossary();
