const header = document.querySelector("[data-elevate]");
const navLinks = [...document.querySelectorAll(".nav-links a")];

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
        link.classList.toggle("is-current", link.getAttribute("href") === `#${id}`);
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

copyButton?.addEventListener("click", async () => {
  if (!bib) return;
  const original = copyButton.textContent;
  const text = bib.textContent.trim();
  const fallbackCopy = () => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    return copied;
  };

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else if (!fallbackCopy()) {
      throw new Error("copy failed");
    }
    copyButton.textContent = "Copied";
  } catch {
    copyButton.textContent = fallbackCopy() ? "Copied" : "Select BibTeX";
  }
  window.setTimeout(() => {
    copyButton.textContent = original;
  }, 1600);
});
