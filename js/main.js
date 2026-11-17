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
  'Naïve Sum (256 frames)',
  'CNN',
  'U-Net',
  'Attn-Res U-Net (Proposed ⭐)'
];

// ─── Experiment Results (three-scene test averages) ───
const data = {
  psnr: {
    values:  [20.955, 26.517, 27.493, 29.10],
    label:   'PSNR (dB)',
    color:   'rgba(0, 212, 255, 0.85)',
    border:  'rgba(0, 212, 255, 1)',
    yLabel:  'PSNR (dB)',
  },
  ssim: {
    values:  [0.2924, 0.7607, 0.7863, 0.8400],
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
          label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(activeMetric === 'ssim' ? 4 : 2)}${activeMetric === 'psnr' ? ' dB' : ''}`
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

/* ─── 6. Scene-aware interactive image comparison ─── */
const sceneDatabase = {
  attic: {
    name: 'attic',
    images: {
      gt: 'assets/results/attic_gt.png',
      naive: 'assets/results/attic_naivesum.png',
      cnn: 'assets/results/attic_cnn.png',
      unet: 'assets/results/attic_unet.png',
    },
    metrics: {
      gt: { psnr: '∞', ssim: '1.0000' },
      naive: { psnr: '21.769', ssim: '0.3077' },
      cnn: { psnr: '28.41', ssim: '0.7470' },
      unet: { psnr: '29.82', ssim: '0.7940' },
      attnres: { psnr: '—', ssim: '—' },
    }
  },
  bachelors: {
    name: 'bachelors-quarters',
    images: {
      gt: 'assets/results/bachelors_gt.png',
      naive: 'assets/results/bachelors_naivesum.png',
      cnn: 'assets/results/bachelors_cnn.png',
      unet: 'assets/results/bachelors_unet.png',
    },
    metrics: {
      gt: { psnr: '∞', ssim: '1.0000' },
      naive: { psnr: '22.152', ssim: '0.2611' },
      cnn: { psnr: '27.48', ssim: '0.8110' },
      unet: { psnr: '28.67', ssim: '0.8310' },
      attnres: { psnr: '—', ssim: '—' },
    }
  },
  barbershop: {
    name: 'barbershop',
    images: {
      gt: 'assets/results/barbershop_gt.png',
      naive: 'assets/results/barbershop_naivesum.png',
      cnn: 'assets/results/barbershop_cnn.png',
      unet: 'assets/results/barbershop_unet.png',
    },
    metrics: {
      gt: { psnr: '∞', ssim: '1.0000' },
      naive: { psnr: '18.943', ssim: '0.3085' },
      cnn: { psnr: '23.66', ssim: '0.7240' },
      unet: { psnr: '23.99', ssim: '0.7340' },
      attnres: { psnr: '—', ssim: '—' },
    }
  }
};

const archLabels = {
  gt:          'Ground Truth (Clean PNG)',
  naive:       'Naïve Sum (256 frames)',
  cnn:         'CNN',
  unet:        'U-Net',
  attnres:     'Attn-Res U-Net (Proposed ⭐)',
};

function updateCompare() {
  const sceneKey = document.getElementById('sceneSelect') ? document.getElementById('sceneSelect').value : 'attic';
  const lKey = document.getElementById('leftArch').value;
  const rKey = document.getElementById('rightArch').value;

  const currentScene = sceneDatabase[sceneKey] || sceneDatabase.attic;

  // Resolve left image
  let leftSrc = (currentScene.images && currentScene.images[lKey]) ? currentScene.images[lKey] : 'assets/placeholder_gen.svg';
  let leftM = (currentScene.metrics && currentScene.metrics[lKey]) ? currentScene.metrics[lKey] : { psnr: '—', ssim: '—' };

  // Resolve right image
  let rightSrc = (currentScene.images && currentScene.images[rKey]) ? currentScene.images[rKey] : 'assets/placeholder_gen.svg';
  let rightM = (currentScene.metrics && currentScene.metrics[rKey]) ? currentScene.metrics[rKey] : { psnr: '—', ssim: '—' };

  document.getElementById('leftImg').src = leftSrc;
  document.getElementById('leftImg').alt = archLabels[lKey] + ' output';
  document.getElementById('leftLabel').textContent = archLabels[lKey];
  document.getElementById('leftPSNR').textContent  = 'PSNR: ' + leftM.psnr + (leftM.psnr !== '—' && leftM.psnr !== '∞' ? ' dB' : (leftM.psnr === '∞' ? ' dB' : ''));
  document.getElementById('leftSSIM').textContent  = 'SSIM: ' + leftM.ssim;

  document.getElementById('rightImg').src = rightSrc;
  document.getElementById('rightImg').alt = archLabels[rKey] + ' output';
  document.getElementById('rightLabel').textContent = archLabels[rKey];
  document.getElementById('rightPSNR').textContent  = 'PSNR: ' + rightM.psnr + (rightM.psnr !== '—' && rightM.psnr !== '∞' ? ' dB' : (rightM.psnr === '∞' ? ' dB' : ''));
  document.getElementById('rightSSIM').textContent  = 'SSIM: ' + rightM.ssim;

}

if (document.getElementById('sceneSelect')) {
  document.getElementById('sceneSelect').addEventListener('change', updateCompare);
}
document.getElementById('leftArch').addEventListener('change',  updateCompare);
document.getElementById('rightArch').addEventListener('change', updateCompare);
updateCompare();  // initial render

/* ─── 7. Low-light Exposure Booster ─── */
const exposureFilters = {
  normal: 'none',
  boost: 'brightness(1.35) contrast(1.15)',
  high: 'brightness(1.6) contrast(1.3)'
};
document.querySelectorAll('.exposure-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.exposure-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filterVal = exposureFilters[btn.dataset.exp] || 'none';
    document.querySelectorAll('.compare-img').forEach(img => {
      img.style.filter = filterVal;
    });
  });
});

/* ─── 8. Scroll-reveal animation ─── */
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
