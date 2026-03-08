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

// ───── Hero Probe Card Cycling ─────
const probeData = [
  { a: '"Companies in the AI industry"', b: '"Companies NOT in the AI industry"', sim: 0.877, cat: "Negation", verdict: "Collapse" },
  { a: '"Acquired by their former subsidiaries"', b: '"Subsidiaries acquired by parent companies"', sim: 0.951, cat: "Compositional", verdict: "Collapse" },
  { a: '"Companies with exactly 50 employees"', b: '"Companies with 11-200 employees"', sim: 0.861, cat: "Numeric Precision", verdict: "Collapse" },
  { a: '"Events before 2020"', b: '"Events after 2020"', sim: 0.822, cat: "Temporal", verdict: "High Sim" },
  { a: '"Cheap luxury hotels"', b: '"Affordable premium hotels"', sim: 0.866, cat: "Antonym Confusion", verdict: "Collapse" },
  { a: '"Python AND NOT JavaScript"', b: '"Python OR JavaScript"', sim: 0.607, cat: "Boolean Logic", verdict: "Moderate" },
];

const probeEls = {
  queryA: document.getElementById("probeQueryA"),
  queryB: document.getElementById("probeQueryB"),
  fill: document.getElementById("probeSimFill"),
  value: document.getElementById("probeSimValue"),
  cat: document.getElementById("probeCat"),
  verdict: document.getElementById("probeVerdict"),
};

let probeIndex = 0;
function cycleProbe() {
  const d = probeData[probeIndex];
  // Fade out
  probeEls.queryA.style.opacity = "0";
  probeEls.queryB.style.opacity = "0";
  probeEls.fill.style.width = "0%";

  setTimeout(() => {
    probeEls.queryA.textContent = d.a;
    probeEls.queryB.textContent = d.b;
    probeEls.value.textContent = d.sim.toFixed(3);
    probeEls.cat.textContent = d.cat;
    probeEls.verdict.textContent = d.verdict;
    // Fade in
    probeEls.queryA.style.opacity = "1";
    probeEls.queryB.style.opacity = "1";
    setTimeout(() => {
      probeEls.fill.style.width = (d.sim * 100).toFixed(1) + "%";
    }, 100);
  }, 300);

  probeIndex = (probeIndex + 1) % probeData.length;
}

// Start first probe immediately, then cycle
cycleProbe();
setInterval(cycleProbe, 3500);

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
const heroStats = document.querySelector(".hero-stats-bar");
if (heroStats) heroObserver.observe(heroStats);

// ───── Reveal Observer ─────
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
);
document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

// ───── Number Highlights ─────
const numObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.querySelectorAll(".num").forEach((n, i) => {
          setTimeout(() => n.classList.add("lit"), i * 120);
        });
        numObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.2 }
);
document.querySelectorAll(".pipeline-step, .arch-card").forEach((el) => numObserver.observe(el));

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

// ───── 3D Tilt Cards ─────
document.querySelectorAll(".tilt-card").forEach((card) => {
  card.addEventListener("mousemove", (e) => {
    const rect = card.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    card.style.transform = `perspective(600px) rotateY(${x * 6}deg) rotateX(${-y * 6}deg)`;
  });
  card.addEventListener("mouseleave", () => {
    card.style.transform = "";
  });
});

// ───── Scroll-Driven Pipeline Fill ─────
const pipeline = document.getElementById("pipeline");
const pipelineFill = document.getElementById("pipelineFill");
const pipelineSteps = pipeline ? pipeline.querySelectorAll(".pipeline-step") : [];

function updatePipeline() {
  if (!pipeline || !pipelineFill) return;
  const rect = pipeline.getBoundingClientRect();
  const viewH = window.innerHeight;

  // How far through the pipeline section we've scrolled
  const start = rect.top - viewH * 0.7;
  const end = rect.bottom - viewH * 0.3;
  const progress = Math.max(0, Math.min(1, -start / (end - start)));

  pipelineFill.style.height = (progress * 100).toFixed(1) + "%";

  // Activate steps as we scroll past them
  pipelineSteps.forEach((step, i) => {
    const stepProgress = (i + 1) / pipelineSteps.length;
    step.classList.toggle("active", progress >= stepProgress * 0.85);
  });
}

if (pipeline) {
  window.addEventListener("scroll", updatePipeline, { passive: true });
  updatePipeline();
}

// ───── CTA Typing Effect ─────
const ctaCodeText = document.getElementById("ctaCodeText");
const ctaCommand = "pip install searchprobe && searchprobe run --providers exa,tavily,brave";

function typeCommand() {
  if (!ctaCodeText) return;
  let i = 0;
  ctaCodeText.textContent = "";
  function type() {
    if (i < ctaCommand.length) {
      ctaCodeText.textContent += ctaCommand[i];
      i++;
      setTimeout(type, 28 + Math.random() * 32);
    }
  }
  type();
}

const ctaObserver = new IntersectionObserver(
  (entries) => {
    if (entries[0].isIntersecting) {
      typeCommand();
      ctaObserver.disconnect();
    }
  },
  { threshold: 0.4 }
);
const ctaCode = document.getElementById("ctaCode");
if (ctaCode) ctaObserver.observe(ctaCode);

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
