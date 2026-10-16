const header = document.querySelector("[data-elevate]");
const navLinks = [...document.querySelectorAll(".nav-links a")];

if (header && "ResizeObserver" in window) {
  new ResizeObserver(() => {
    document.documentElement.style.setProperty("--header-height", `${header.offsetHeight}px`);
  }).observe(header);
}

const updateHeader = () => {
  header?.classList.toggle("is-elevated", window.scrollY > 8);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

const sectionObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const id = entry.target.id;
      navLinks.forEach(link => {
        const current = link.getAttribute("href") === `#${id}`;
        link.classList.toggle("is-current", current);
        if (current) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    });
  },
  { rootMargin: "-34% 0px -58% 0px", threshold: 0.01 }
);

navLinks
  .map(link => link.getAttribute("href"))
  .filter(href => href?.startsWith("#"))
  .map(href => href.slice(1))
  .forEach(id => {
  const section = document.getElementById(id);
  if (section) sectionObserver.observe(section);
});

const copyButton = document.querySelector("[data-copy-bib]");
const bib = document.querySelector("[data-bibtex]");
const copyStatus = document.querySelector("[data-copy-status]");
let copyReset;

copyButton?.addEventListener("click", async () => {
  if (!bib) return;
  const text = bib.textContent.trim();
  const fallbackCopy = () => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      textarea.remove();
      copyButton.focus({ preventScroll: true });
    }
  };
  let copied = false;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      copied = true;
    } else copied = fallbackCopy();
  } catch {
    copied = fallbackCopy();
  }
  if (!copied) {
    const range = document.createRange();
    range.selectNodeContents(bib);
    const selection = window.getSelection();
    selection?.removeAllRanges();
    selection?.addRange(range);
  }
  copyButton.textContent = copied ? "Copied" : "BibTeX selected";
  if (copyStatus) copyStatus.textContent = copied ? "BibTeX copied to clipboard." : "BibTeX selected. Press Ctrl+C or Command+C to copy.";
  window.clearTimeout(copyReset);
  copyReset = window.setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
  }, 1600);
});

const figureDialog = document.querySelector("[data-figure-dialog]");
const figurePreview = document.querySelector("[data-figure-preview]");
let figureTrigger;

if (figureDialog && figurePreview && typeof figureDialog.showModal === "function") {
  document.querySelectorAll(".paper-figure > img").forEach(img => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "figure-zoom";
    button.setAttribute("aria-label", `Enlarge figure: ${img.alt}`);
    button.setAttribute("aria-haspopup", "dialog");
    img.before(button);
    button.appendChild(img);
    const hint = document.createElement("span");
    hint.className = "figure-zoom-hint";
    hint.textContent = "Enlarge ↗";
    hint.setAttribute("aria-hidden", "true");
    button.appendChild(hint);
    button.addEventListener("click", () => {
      figureTrigger = button;
      figurePreview.src = img.currentSrc || img.src;
      figurePreview.alt = img.alt;
      figureDialog.showModal();
      document.body.classList.add("figure-is-open");
    });
  });
  figureDialog.querySelector("[data-close-figure]")?.addEventListener("click", () => figureDialog.close());
  figureDialog.addEventListener("click", event => {
    const rect = figureDialog.getBoundingClientRect();
    if (event.target === figureDialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) {
      figureDialog.close();
    }
  });
  figureDialog.addEventListener("close", () => {
    document.body.classList.remove("figure-is-open");
    figurePreview.removeAttribute("src");
    figureTrigger?.focus({ preventScroll: true });
  });
}
