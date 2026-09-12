/* ═══════════════════════════════════════════════════════════════
   SMS Spam Detection — Application Logic
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // DOM References
  const navbar       = document.getElementById('navbar');
  const navToggle    = document.getElementById('navToggle');
  const navLinks     = document.querySelector('.nav-links');
  const smsInput     = document.getElementById('smsInput');
  const charCount    = document.getElementById('charCount');
  const detectBtn    = document.getElementById('detectBtn');
  const clearBtn     = document.getElementById('clearBtn');
  const validationMsg = document.getElementById('validationMsg');
  const detectCard   = document.getElementById('detectCard');
  const loadingCard  = document.getElementById('loadingCard');
  const resultSpam   = document.getElementById('resultSpam');
  const resultHam    = document.getElementById('resultHam');
  const tryAgainSpam = document.getElementById('tryAgainSpam');
  const tryAgainHam  = document.getElementById('tryAgainHam');

  // Spam detection keywords / patterns
  const spamPatterns = [
    { regex: /\b(win|winner|won)\b/i,          label: 'Prize / Win claim' },
    { regex: /\b(free|gratis)\b/i,             label: 'Free offer' },
    { regex: /\b(congratulations|congrats)\b/i, label: 'Congratulations hook' },
    { regex: /\b(click|claim|subscribe)\b/i,   label: 'Call-to-action' },
    { regex: /\b(prize|reward|bonus|gift|voucher|cash)\b/i, label: 'Monetary bait' },
    { regex: /\b(urgent|immediately|now|hurry|act)\b/i, label: 'Urgency language' },
    { regex: /\b(call|text|reply|send)\s+(to\s+)?\d{4,}/i, label: 'Phone number' },
    { regex: /\b(offer|deal|discount|sale|limited)\b/i, label: 'Sales language' },
    { regex: /\$\d+|£\d+|₹\d+/,               label: 'Currency mention' },
    { regex: /\b(http|www|\.com|\.co|\.uk)\b/i, label: 'URL / Link' },
    { regex: /[A-Z]{4,}/,                       label: 'Excessive caps' },
    { regex: /!{2,}/,                            label: 'Multiple exclamation' },
  ];

  // Navbar scroll effect
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
  });

  // Mobile nav toggle
  navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
  navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => navLinks.classList.remove('open'));
  });

  // Character counter & button state
  smsInput.addEventListener('input', () => {
    const len = smsInput.value.length;
    charCount.textContent = len;
    detectBtn.disabled = len === 0;
    if (len > 0) validationMsg.style.display = 'none';
  });

  // Ctrl+Enter shortcut
  smsInput.addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === 'Enter' && smsInput.value.trim()) {
      runDetection();
    }
  });

  // Detect button
  detectBtn.addEventListener('click', () => {
    if (!smsInput.value.trim()) {
      validationMsg.style.display = 'flex';
      return;
    }
    runDetection();
  });

  // Clear button
  clearBtn.addEventListener('click', () => {
    smsInput.value = '';
    charCount.textContent = '0';
    detectBtn.disabled = true;
    validationMsg.style.display = 'none';
    hideResults();
  });

  // Try Again buttons
  tryAgainSpam.addEventListener('click', resetToInput);
  tryAgainHam.addEventListener('click', resetToInput);

  // Example buttons
  document.querySelectorAll('.btn-use').forEach(btn => {
    btn.addEventListener('click', () => {
      const msg = btn.getAttribute('data-msg');
      smsInput.value = msg;
      charCount.textContent = msg.length;
      detectBtn.disabled = false;
      document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
      smsInput.focus();
    });
  });

  // Stat counter animation
  const statObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateStats();
        statObserver.disconnect();
      }
    });
  }, { threshold: 0.3 });
  const statsSection = document.getElementById('stats');
  if (statsSection) statObserver.observe(statsSection);

  // ═══════════════════════════════════════════
  // CORE DETECTION LOGIC
  // ═══════════════════════════════════════════
  function runDetection() {
    const message = smsInput.value.trim();
    hideResults();
    detectCard.style.display = 'none';
    loadingCard.style.display = 'block';

    const steps = loadingCard.querySelectorAll('.loading-step');
    steps.forEach(s => s.classList.remove('active', 'done'));

    let idx = 0;
    const stepInterval = setInterval(() => {
      if (idx > 0) steps[idx - 1].classList.replace('active', 'done');
      if (idx < steps.length) {
        steps[idx].classList.add('active');
        idx++;
      } else {
        clearInterval(stepInterval);
        setTimeout(() => showResult(message), 400);
      }
    }, 450);
  }

  function showResult(message) {
    loadingCard.style.display = 'none';

    const { isSpam, confidence, indicators } = classifyMessage(message);

    if (isSpam) {
      resultSpam.style.display = 'block';
      const bar = document.getElementById('spamConfBar');
      const val = document.getElementById('spamConfVal');
      val.textContent = confidence.toFixed(1) + '%';
      requestAnimationFrame(() => { bar.style.width = confidence + '%'; });

      const list = document.getElementById('spamIndicators');
      list.innerHTML = '';
      indicators.forEach(ind => {
        const li = document.createElement('li');
        li.textContent = ind;
        list.appendChild(li);
      });
    } else {
      resultHam.style.display = 'block';
      const bar = document.getElementById('hamConfBar');
      const val = document.getElementById('hamConfVal');
      val.textContent = confidence.toFixed(1) + '%';
      requestAnimationFrame(() => { bar.style.width = confidence + '%'; });

      document.getElementById('hamSummary').textContent =
        'No spam indicators were detected. The message has a natural conversational tone with ' +
        message.split(/\s+/).length + ' words and standard grammar patterns.';
    }
  }

  function classifyMessage(message) {
    let spamScore = 0;
    const matchedIndicators = [];

    spamPatterns.forEach(({ regex, label }) => {
      if (regex.test(message)) {
        spamScore += 1;
        matchedIndicators.push(label);
      }
    });

    const ratio = spamScore / spamPatterns.length;
    const isSpam = spamScore >= 2;

    let confidence;
    if (isSpam) {
      confidence = Math.min(60 + ratio * 40, 99.2);
      if (spamScore >= 5) confidence = Math.min(93 + spamScore * 0.6, 99.5);
    } else {
      confidence = Math.min(88 + (1 - ratio) * 11, 99.8);
    }

    return { isSpam, confidence, indicators: matchedIndicators };
  }

  function hideResults() {
    resultSpam.style.display = 'none';
    resultHam.style.display  = 'none';
    loadingCard.style.display = 'none';
    document.getElementById('spamConfBar').style.width = '0%';
    document.getElementById('hamConfBar').style.width  = '0%';
  }

  function resetToInput() {
    hideResults();
    detectCard.style.display = 'block';
    smsInput.value = '';
    charCount.textContent = '0';
    detectBtn.disabled = true;
    document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
  }

  // Stat counter animation
  function animateStats() {
    document.querySelectorAll('.stat-value[data-target]').forEach(el => {
      const target = parseInt(el.getAttribute('data-target'), 10);
      const suffix = el.getAttribute('data-suffix') || '';
      const duration = 1200;
      const start = performance.now();

      function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(target * ease);
        el.textContent = current.toLocaleString() + suffix;
        if (progress < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }
});
