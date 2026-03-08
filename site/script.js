// ───── Nav scroll ─────
const nav = document.getElementById("nav");
window.addEventListener("scroll", () => {
  nav.classList.toggle("scrolled", window.scrollY > 20);
});

// ───── Mobile nav ─────
const navToggle = document.getElementById("navToggle");
const navLinks = document.querySelector(".nav-links");
navToggle.addEventListener("click", () => navLinks.classList.toggle("open"));
navLinks.querySelectorAll("a").forEach((a) => {
  a.addEventListener("click", () => navLinks.classList.remove("open"));
});

// ───── Counter animation ─────
function animateCounters() {
  document.querySelectorAll("[data-target]").forEach((el) => {
    const target = parseInt(el.dataset.target, 10);
    const duration = 1600;
    const start = performance.now();

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(eased * target).toLocaleString();
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

// ───── Intersection Observer ─────
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
);

// Apply fade-up to content blocks
document.querySelectorAll(
  ".problem-card, .pipeline-step, .results-block, .finding-card, .arch-card, .geometry-table-wrap, .section-cta"
).forEach((el, i) => {
  el.classList.add("fade-up");
  // Stagger siblings of the same type
  const siblings = el.parentElement.querySelectorAll(`:scope > .${el.className.split(" ")[0]}`);
  const idx = Array.from(siblings).indexOf(el);
  if (idx >= 0 && idx < 6) el.classList.add(`stagger-${idx + 1}`);
  observer.observe(el);
});

// ───── Hero counters on view ─────
let countersStarted = false;
const heroObserver = new IntersectionObserver(
  (entries) => {
    if (entries[0].isIntersecting && !countersStarted) {
      countersStarted = true;
      animateCounters();
    }
  },
  { threshold: 0.3 }
);
const heroStats = document.querySelector(".hero-stats");
if (heroStats) heroObserver.observe(heroStats);

// ───── Provider bar animation ─────
const barsObserver = new IntersectionObserver(
  (entries) => {
    if (entries[0].isIntersecting) {
      document.querySelectorAll(".bar-fill").forEach((bar, i) => {
        setTimeout(() => {
          bar.style.width = bar.style.getPropertyValue("--bar-width");
        }, i * 80);
      });
      barsObserver.disconnect();
    }
  },
  { threshold: 0.2 }
);
document.querySelectorAll(".bar-fill").forEach((bar) => {
  bar.style.width = "0%";
});
const providerBars = document.querySelector(".provider-bars");
if (providerBars) barsObserver.observe(providerBars);

// ───── Category bar animation ─────
const catObserver = new IntersectionObserver(
  (entries) => {
    if (entries[0].isIntersecting) {
      document.querySelectorAll(".cat-fill").forEach((bar, i) => {
        setTimeout(() => {
          bar.style.width = bar.style.getPropertyValue("--cat-width");
        }, i * 50);
      });
      catObserver.disconnect();
    }
  },
  { threshold: 0.2 }
);
const catBars = document.querySelector(".category-bars");
if (catBars) catObserver.observe(catBars);

// ───── Smooth anchors ─────
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", (e) => {
    const target = document.querySelector(anchor.getAttribute("href"));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
});
