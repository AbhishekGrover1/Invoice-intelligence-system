/**
 * Invoice Intelligence System — frontend logic.
 * No build step, no framework: DOM APIs + fetch() against the FastAPI
 * backend that serves this same page (relative /api/* URLs).
 */
(() => {
  "use strict";

  const $ = (sel, ctx = document) => ctx.querySelector(sel);
  const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------------------------
   * Footer year
   * ------------------------------------------------------------------- */
  const yearEl = $("#year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* ---------------------------------------------------------------------
   * Mobile nav
   * ------------------------------------------------------------------- */
  const navToggle = $("#navToggle");
  const navMobile = $("#navMobile");
  if (navToggle && navMobile) {
    navToggle.addEventListener("click", () => {
      const isOpen = navMobile.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(isOpen));
      $("#navToggleIcon").innerHTML = isOpen
        ? '<path d="M5 5l14 14M19 5 5 19"/>'
        : '<path d="M4 7h16M4 12h16M4 17h16"/>';
    });
    $$("#navMobile a").forEach((a) =>
      a.addEventListener("click", () => {
        navMobile.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
        $("#navToggleIcon").innerHTML = '<path d="M4 7h16M4 12h16M4 17h16"/>';
      })
    );
  }

  /* ---------------------------------------------------------------------
   * Reveal-on-scroll
   * ------------------------------------------------------------------- */
  const revealEls = $$(".reveal");
  if (revealEls.length) {
    if (prefersReducedMotion || !("IntersectionObserver" in window)) {
      revealEls.forEach((el) => el.classList.add("is-visible"));
    } else {
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add("is-visible");
              io.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
      );
      revealEls.forEach((el) => io.observe(el));
    }
  }

  /* ---------------------------------------------------------------------
   * Hero stat count-up
   * ------------------------------------------------------------------- */
  function animateCountUp(el) {
    const target = parseFloat(el.dataset.countup);
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const suffix = el.dataset.suffix || "";
    if (Number.isNaN(target)) return;

    if (prefersReducedMotion) {
      el.textContent = target.toFixed(decimals) + suffix;
      return;
    }

    const duration = 1400;
    const start = performance.now();

    function tick(now) {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
      const value = target * eased;
      el.textContent = value.toFixed(decimals) + suffix;
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  const countEls = $$("[data-countup]");
  if (countEls.length) {
    if (!("IntersectionObserver" in window)) {
      countEls.forEach(animateCountUp);
    } else {
      const countIo = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              animateCountUp(entry.target);
              countIo.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.5 }
      );
      countEls.forEach((el) => countIo.observe(el));
    }
  }

  /* ---------------------------------------------------------------------
   * Generic tab switcher — works for both the freight/risk demo tabs and
   * the single/batch sub-tabs, keyed by a data attribute name.
   * ------------------------------------------------------------------- */
  function wireTabs(triggerAttr, panelResolver) {
    const triggers = $$(`[${triggerAttr}]`);
    triggers.forEach((trigger) => {
      trigger.addEventListener("click", () => {
        const key = trigger.getAttribute(triggerAttr);
        triggers.forEach((t) => {
          const active = t === trigger;
          t.classList.toggle("is-active", active);
          t.setAttribute("aria-selected", String(active));
        });
        panelResolver(key);
      });
    });
  }

  wireTabs("data-demo-tab", (key) => {
    $("#panel-freight").classList.toggle("is-active", key === "freight");
    $("#panel-risk").classList.toggle("is-active", key === "risk");
  });

  wireTabs("data-risk-tab", (key) => {
    $("#risk-single").classList.toggle("is-active", key === "single");
    $("#risk-batch").classList.toggle("is-active", key === "batch");
  });

  /* ---------------------------------------------------------------------
   * Shared helpers
   * ------------------------------------------------------------------- */
  function setFieldError(fieldId, hasError) {
    const field = document.getElementById(fieldId);
    if (field) field.classList.toggle("has-error", hasError);
  }

  function retrigger(el, className) {
    el.classList.remove(className);
    void el.offsetWidth; // force reflow so the animation restarts
    el.classList.add(className);
  }

  async function parseApiError(response) {
    try {
      const body = await response.json();
      if (typeof body.detail === "string") return body.detail;
      if (Array.isArray(body.detail)) {
        return body.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
      }
      return `Request failed (${response.status}).`;
    } catch {
      return `Request failed (${response.status}).`;
    }
  }

  /* ---------------------------------------------------------------------
   * Freight Cost predictor
   * ------------------------------------------------------------------- */
  const freightForm = $("#freightForm");
  if (freightForm) {
    const dollarsInput = $("#freightDollars");
    const idle = $("#freightIdle");
    const loading = $("#freightLoading");
    const errorEl = $("#freightError");
    const resultEl = $("#freightResult");
    const valueEl = $("#freightValue");
    const subEl = $("#freightSub");
    const barFill = $("#freightBarFill");
    const submitBtn = $("#freightSubmit");

    freightForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const dollars = parseFloat(dollarsInput.value);
      const valid = Number.isFinite(dollars) && dollars > 0;
      setFieldError("freightDollarsField", !valid);
      if (!valid) return;

      idle.style.display = "none";
      resultEl.classList.remove("is-active");
      errorEl.classList.remove("is-active");
      loading.classList.add("is-active");
      submitBtn.disabled = true;

      try {
        const res = await fetch("/api/predict/freight-cost", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dollars }),
        });
        if (!res.ok) throw new Error(await parseApiError(res));
        const data = await res.json();

        valueEl.textContent = `$${data.predicted_freight.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        subEl.textContent = `≈ ${data.freight_ratio_pct.toFixed(2)}% of the $${data.dollars.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} invoice value`;
        // Visual scaling only (×6, floor 4%) so small real-world ratios are
        // still legible as a bar — the exact percentage is in the text above.
        const barWidth = Math.min(100, Math.max(4, data.freight_ratio_pct * 6));
        barFill.style.width = "0%";
        requestAnimationFrame(() => { barFill.style.width = `${barWidth}%`; });

        loading.classList.remove("is-active");
        resultEl.classList.add("is-active");
      } catch (err) {
        loading.classList.remove("is-active");
        errorEl.textContent = err.message || "Something went wrong reaching the API.";
        errorEl.classList.add("is-active");
      } finally {
        submitBtn.disabled = false;
      }
    });
  }

  /* ---------------------------------------------------------------------
   * Invoice Risk — single prediction
   * ------------------------------------------------------------------- */
  const riskForm = $("#riskForm");
  if (riskForm) {
    const fields = {
      riskQuantity: { min: 1, integer: true },
      riskDollars: { min: 0.01, integer: false },
      riskFreight: { min: 0, integer: false },
      riskTotalQty: { min: 1, integer: true },
      riskTotalDollars: { min: 0.01, integer: false },
    };

    const idle = $("#riskIdle");
    const loading = $("#riskLoading");
    const errorEl = $("#riskError");
    const stampResult = $("#riskStampResult");
    const stampEl = $("#riskStamp");
    const probPct = $("#riskProbPct");
    const probFill = $("#riskProbFill");
    const noteEl = $("#riskNote");
    const submitBtn = $("#riskSubmit");

    function readValidated() {
      const values = {};
      let allValid = true;
      for (const [id, rule] of Object.entries(fields)) {
        const input = document.getElementById(id);
        const raw = parseFloat(input.value);
        const valid = Number.isFinite(raw) && raw >= rule.min && (!rule.integer || Number.isInteger(raw));
        setFieldError(`${id}Field`, !valid);
        if (!valid) allValid = false;
        values[id] = raw;
      }
      return allValid ? values : null;
    }

    riskForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const values = readValidated();
      if (!values) return;

      idle.style.display = "none";
      stampResult.classList.remove("is-active");
      errorEl.classList.remove("is-active");
      loading.classList.add("is-active");
      submitBtn.disabled = true;

      try {
        const res = await fetch("/api/predict/invoice-risk", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            invoice_quantity: values.riskQuantity,
            invoice_dollars: values.riskDollars,
            freight: values.riskFreight,
            total_item_quantity: values.riskTotalQty,
            total_item_dollars: values.riskTotalDollars,
          }),
        });
        if (!res.ok) throw new Error(await parseApiError(res));
        const data = await res.json();

        renderStamp(data);

        loading.classList.remove("is-active");
        stampResult.classList.add("is-active");
      } catch (err) {
        loading.classList.remove("is-active");
        errorEl.textContent = err.message || "Something went wrong reaching the API.";
        errorEl.classList.add("is-active");
      } finally {
        submitBtn.disabled = false;
      }
    });

    function renderStamp(data) {
      stampEl.classList.remove("stamp--safe", "stamp--risk", "is-stamped");
      probFill.classList.remove("stamp-result__prob-fill--safe", "stamp-result__prob-fill--risk");

      if (data.flagged) {
        stampEl.classList.add("stamp--risk");
        stampEl.innerHTML = "Flagged<br>for Review";
        probFill.classList.add("stamp-result__prob-fill--risk");
        noteEl.textContent = "This invoice's profile resembles cases in the training data with a price gap or receiving delay worth a second look.";
      } else {
        stampEl.classList.add("stamp--safe");
        stampEl.innerHTML = "Auto<br>Approved";
        probFill.classList.add("stamp-result__prob-fill--safe");
        noteEl.textContent = "This invoice's profile matches invoices that were historically paid without needing manual review.";
      }

      probPct.textContent = `${(data.flag_probability * 100).toFixed(1)}%`;
      probFill.style.width = "0%";
      requestAnimationFrame(() => { probFill.style.width = `${data.flag_probability * 100}%`; });

      requestAnimationFrame(() => retrigger(stampEl, "is-stamped"));
    }
  }

  /* ---------------------------------------------------------------------
   * Invoice Risk — batch CSV upload
   * ------------------------------------------------------------------- */
  const batchDropzone = $("#batchDropzone");
  if (batchDropzone) {
    const fileInput = $("#batchFileInput");
    const fileNameEl = $("#batchFileName");
    const analyzeBtn = $("#batchAnalyzeBtn");
    const errorEl = $("#batchError");
    const summaryEl = $("#batchSummary");
    const totalEl = $("#batchTotal");
    const flaggedEl = $("#batchFlagged");
    const approvedEl = $("#batchApproved");
    const tableWrap = $("#batchTableWrap");
    const tableBody = $("#batchTableBody");

    let selectedFile = null;

    function setFile(file) {
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".csv")) {
        errorEl.textContent = "Please choose a .csv file.";
        errorEl.classList.add("is-active");
        return;
      }
      selectedFile = file;
      fileNameEl.textContent = file.name;
      analyzeBtn.disabled = false;
      errorEl.classList.remove("is-active");
    }

    batchDropzone.addEventListener("click", () => fileInput.click());
    batchDropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    });
    fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

    ["dragenter", "dragover"].forEach((evt) =>
      batchDropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        batchDropzone.classList.add("is-dragover");
      })
    );
    ["dragleave", "drop"].forEach((evt) =>
      batchDropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        batchDropzone.classList.remove("is-dragover");
      })
    );
    batchDropzone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      setFile(file);
    });

    analyzeBtn.addEventListener("click", async () => {
      if (!selectedFile) return;

      errorEl.classList.remove("is-active");
      summaryEl.classList.remove("is-active");
      tableWrap.classList.remove("is-active");
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Analyzing…";

      try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const res = await fetch("/api/predict/invoice-risk/batch", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error(await parseApiError(res));
        const data = await res.json();

        totalEl.textContent = data.total_rows;
        flaggedEl.textContent = data.flagged_count;
        approvedEl.textContent = data.auto_approved_count;
        summaryEl.classList.add("is-active");

        tableBody.innerHTML = data.results
          .map(
            (r) => `
          <tr>
            <td>${r.row}</td>
            <td><span class="pill ${r.flagged ? "pill--risk" : "pill--safe"}">${r.flagged ? "Flagged" : "Approved"}</span></td>
            <td>${(r.flag_probability * 100).toFixed(1)}%</td>
            <td>${r.confidence_pct.toFixed(1)}%</td>
          </tr>`
          )
          .join("");
        tableWrap.classList.add("is-active");
      } catch (err) {
        errorEl.textContent = err.message || "Something went wrong reaching the API.";
        errorEl.classList.add("is-active");
      } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze Batch";
      }
    });
  }
})();
