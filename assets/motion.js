<script>
(() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  // count numbers up from 0, keeping prefix, suffix, decimals and thousand separators
  const countUp = el => {
    const m = el.textContent.trim().match(/^([^\d]*)([\d,]+(?:\.\d+)?)(.*)$/);
    if (!m || reduce) return;
    const [, pre, num, post] = m, target = parseFloat(num.replace(/,/g, '')), dec = (num.split('.')[1] || '').length;
    const comma = num.includes(','), t0 = performance.now(), dur = 1400;
    const fmt = v => { let s = v.toFixed(dec); if (comma) s = Number(s).toLocaleString('en-US'); return pre + s + post; };
    const step = t => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(target * e); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  };
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add('sm-in');
    // once the entrance is over, drop the reveal classes so hover effects own the transform again
    setTimeout(() => e.target.classList.remove('sm-r', 'sm-in'), 1400);
    e.target.querySelectorAll('.sm-count').forEach(countUp);
    if (e.target.classList.contains('sm-count')) countUp(e.target);
    io.unobserve(e.target);
  }), { threshold: 0.15 });
  const SEL = '.sm-card, .sm-note, .sm-steps li, .sm-fig, .sm-table:not(.small), .sm-stephead, .sm-h3,' +
              '.sm-section:not(.first) > h2, .sm-section:not(.first) > .sm-kicker';
  const prep = root => {
    root.querySelectorAll(SEL).forEach(el => {
      if (el.dataset.smr || el.closest('.sm-hero')) return;
      el.dataset.smr = 1;
      const sib = el.parentElement ? [...el.parentElement.children].indexOf(el) : 0;
      el.style.setProperty('--sd', (Math.min(sib, 4) * 0.09) + 's');
      if (!reduce) el.classList.add('sm-r');
      io.observe(el);
    });
  };
  // hero parallax: sherds follow the pointer, each by its own depth
  document.addEventListener('pointermove', ev => {
    const art = document.querySelector('.sm-hero-art'); if (!art || reduce) return;
    const r = art.getBoundingClientRect();
    art.style.setProperty('--mx', ((ev.clientX - r.left) / r.width - .5).toFixed(3));
    art.style.setProperty('--my', ((ev.clientY - r.top) / r.height - .5).toFixed(3));
  }, { passive: true });
  // the fit preview is a Gradio image: replay its entrance whenever the picture changes
  const watchFit = () => document.querySelectorAll('#sm-fit img').forEach(img => {
    if (img.dataset.smw) return; img.dataset.smw = 1;
    new MutationObserver(() => { img.classList.remove('sm-popin'); void img.offsetWidth; img.classList.add('sm-popin'); })
      .observe(img, { attributes: true, attributeFilter: ['src'] });
    img.classList.add('sm-popin');
  });
  // Gradio renders pages client-side, so keep preparing whatever new content appears
  let queued = false;
  new MutationObserver(() => {
    if (queued) return; queued = true;
    requestAnimationFrame(() => { queued = false; prep(document); watchFit(); });
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
