// ============================================================
//  main.js  —  Single Photon Image Reconstruction Website
// ============================================================

/* ─── 1. NAVBAR scroll effect ─── */
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
});

/* ─── 2. Mobile nav toggle ─── */
document.getElementById('navToggle').addEventListener('click', () => {
  document.getElementById('navLinks').classList.toggle('open');
});
// Close on link click
document.querySelectorAll('.nav-links a').forEach(a => {
  a.addEventListener('click', () => document.getElementById('navLinks').classList.remove('open'));
});

/* ─── 3. Photon Particle Animation ─── */
(function spawnParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  function spawn() {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 4 + 1;
    const x    = Math.random() * 100;
    const delay = Math.random() * 6;
    const dur   = Math.random() * 6 + 4;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${x}%; bottom:${Math.random()*60}%;
      animation-duration:${dur}s; animation-delay:${delay}s;
      background: hsl(${180 + Math.random()*80}, 90%, 70%);
    `;
    container.appendChild(p);
    setTimeout(() => p.remove(), (dur + delay) * 1000 + 500);
  }
  for (let i = 0; i < 40; i++) spawn();
  setInterval(spawn, 300);
})();

/* ─── 4. Architecture Tabs ─── */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + target).classList.add('active');
  });
});

/* ─── 5. Model comparison chart (Chart.js) ─── */
const models = [
  'Naïve Baseline',
  'CNN',
  'U-Net',
  'Attention U-Net',
  'Attn-Res U-Net (Paper ⭐)',
  'Vision Transformer'
];

// ─── UPDATE THESE VALUES with your actual experiment results ───
const data = {
  psnr: {
    values:  [18.5, 22.8, 25.4, 27.2, 28.9, 27.6],  // illustrative dB values
    label:   'PSNR (dB)',
    color:   'rgba(0, 212, 255, 0.85)',
    border:  'rgba(0, 212, 255, 1)',
    yLabel:  'PSNR (dB)',
  },
  ssim: {
    values:  [0.48, 0.63, 0.73, 0.79, 0.84, 0.80],  // illustrative SSIM values
    label:   'SSIM',
    color:   'rgba(124, 58, 237, 0.85)',
    border:  'rgba(124, 58, 237, 1)',
    yLabel:  'SSIM',
  }
};

let activeMetric = 'psnr';

const ctx = document.getElementById('modelChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: models,
    datasets: [{
      label: data[activeMetric].label,
      data:  data[activeMetric].values,
      backgroundColor: data[activeMetric].color,
      borderColor:     data[activeMetric].border,
      borderWidth: 2,
      borderRadius: 6,
      hoverBackgroundColor: data[activeMetric].border,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}${activeMetric === 'psnr' ? ' dB' : ''}`
        },
        backgroundColor: '#111827',
        borderColor: '#1e2d4a',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
      }
    },
    scales: {
      x: {
        ticks: { color: '#94a3b8', font: { family: 'Inter' } },
        grid: { color: 'rgba(255,255,255,0.04)' }
      },
      y: {
        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } },
        grid: { color: 'rgba(255,255,255,0.06)' },
        title: { display: true, text: data[activeMetric].yLabel, color: '#64748b' }
      }
    }
  }
});

function switchMetric(metric) {
  activeMetric = metric;
  const d = data[metric];
  chart.data.datasets[0].data            = d.values;
  chart.data.datasets[0].label           = d.label;
  chart.data.datasets[0].backgroundColor = d.color;
  chart.data.datasets[0].borderColor     = d.border;
  chart.data.datasets[0].hoverBackgroundColor = d.border;
  chart.options.scales.y.title.text      = d.yLabel;
  chart.update();
}

document.querySelectorAll('.chart-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    switchMetric(btn.dataset.metric);
  });
});

/* ─── 6. Architecture image comparison ─── */

// ─── POPULATE THIS with real reconstruction output images ───
// Place images in assets/results/<archname>.png  (or jpg/svg)
// Keys must match the <option value="..."> in the HTML selects.
const archImages = {
  naive:       'assets/placeholder_gen.svg',
  cnn:         'assets/placeholder_gen.svg',
  unet:        'assets/placeholder_gen.svg',
  attnunet:    'assets/placeholder_gen.svg',
  attnres:     'assets/placeholder_gen.svg',
  transformer: 'assets/placeholder_gen.svg',
};

// ─── UPDATE THESE with your real PSNR / SSIM per architecture ───
const archMetrics = {
  naive:       { psnr: '—', ssim: '—' },
  cnn:         { psnr: '—', ssim: '—' },
  unet:        { psnr: '—', ssim: '—' },
  attnunet:    { psnr: '—', ssim: '—' },
  attnres:     { psnr: '—', ssim: '—' },
  transformer: { psnr: '—', ssim: '—' },
};

const archLabels = {
  naive:       'Naïve Baseline',
  cnn:         'CNN',
  unet:        'U-Net',
  attnunet:    'Attention U-Net',
  attnres:     'Attn-Res U-Net (Paper ⭐)',
  transformer: 'Vision Transformer (Ext.)',
};

function updateCompare() {
  const lKey = document.getElementById('leftArch').value;
  const rKey = document.getElementById('rightArch').value;

  document.getElementById('leftImg').src     = archImages[lKey];
  document.getElementById('leftImg').alt     = archLabels[lKey] + ' output';
  document.getElementById('leftLabel').textContent = archLabels[lKey];
  document.getElementById('leftPSNR').textContent  = 'PSNR: ' + archMetrics[lKey].psnr + (archMetrics[lKey].psnr !== '—' ? ' dB' : '');
  document.getElementById('leftSSIM').textContent  = 'SSIM: ' + archMetrics[lKey].ssim;

  document.getElementById('rightImg').src    = archImages[rKey];
  document.getElementById('rightImg').alt    = archLabels[rKey] + ' output';
  document.getElementById('rightLabel').textContent = archLabels[rKey];
  document.getElementById('rightPSNR').textContent  = 'PSNR: ' + archMetrics[rKey].psnr + (archMetrics[rKey].psnr !== '—' ? ' dB' : '');
  document.getElementById('rightSSIM').textContent  = 'SSIM: ' + archMetrics[rKey].ssim;
}

document.getElementById('leftArch').addEventListener('change',  updateCompare);
document.getElementById('rightArch').addEventListener('change', updateCompare);
updateCompare();  // initial render

/* ─── 7. Scroll-reveal animation ─── */
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity    = '1';
      e.target.style.transform  = 'translateY(0)';
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.08 });

document.querySelectorAll(
  '.data-card, .info-card, .step, .arch-tabs, .loss-card, .eval-card, .error-card, .repro-step, .takeaway'
).forEach(el => {
  el.style.opacity   = '0';
  el.style.transform = 'translateY(24px)';
  el.style.transition= 'opacity 0.5s ease, transform 0.5s ease';
  observer.observe(el);
});
