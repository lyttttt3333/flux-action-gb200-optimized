"use strict";
const $ = (id) => document.getElementById(id);
const fmt = (n, digits = 3) => n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
const caseNames = ['将黄色杯中的内容物倒入粉色碗中', '将杯中的内容物倒入碗中'];
let data, abba;

function drawStages(index) {
  const stages = data.history.map(s => ({ ...s, ms: s.cases[index].median_ms }));
  stages.push({ label: '全部无损优化', detail: '最终 ABBA · 固定 launcher', ms: data.cases[index].final_ms });
  $('stage-bars').innerHTML = stages.map((s, i) => `<div class="stage-row ${i === 4 ? 'final' : ''}"><span class="stage-index">0${i + 1}</span><div class="stage-label"><b>${s.label}</b><small>${s.detail}</small></div><div class="stage-track"><div class="stage-fill" style="width:${s.ms / 1500 * 100}%"></div></div><span class="stage-value">${fmt(s.ms, 1)}</span></div>`).join('');
}

function drawAbba(index) {
  const c = abba.cases[index], w = 620, h = 212, left = 40, right = 12, top = 17, bottom = 27;
  const x = i => left + i / 29 * (w - left - right);
  const y = ms => top + (735 - ms) / 60 * (h - top - bottom);
  const grid = [680, 700, 720, 735].map(v => `<line x1="${left}" x2="${w - right}" y1="${y(v)}" y2="${y(v)}" stroke="#263643" stroke-width="1" stroke-dasharray="3 5"/><text x="${left - 9}" y="${y(v) + 3}" fill="#6c8799" font-size="9" text-anchor="end">${v}</text>`).join('');
  const ticks = [0, 9, 19, 29].map(i => `<text x="${x(i)}" y="${h - 7}" fill="#6c8799" font-size="9" text-anchor="middle">${i + 1}</text>`).join('');
  const series = (arr, color, label) => `<polyline points="${arr.map((s, i) => `${x(i)},${y(s * 1000)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="1.8"/>${arr.map((s, i) => `<circle cx="${x(i)}" cy="${y(s * 1000)}" r="2.3" fill="${color}"><title>${label} · 样本 ${i + 1} · ${fmt(s * 1000)} ms</title></circle>`).join('')}`;
  $('abba-chart').innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="输入 ${index + 1} 的 30 次交替测量：基线中位数 ${fmt(c.baseline_median * 1000)} 毫秒，优化后 ${fmt(c.candidate_median * 1000)} 毫秒" font-family="ui-monospace,monospace"><text x="6" y="9" fill="#6c8799" font-size="8">ms</text>${grid}${ticks}${series(c.baseline_seconds, '#76aaff', '固定 Sage2 基线')}${series(c.candidate_seconds, '#78efc0', '全部无损优化')}</svg>`;
}

function drawQuality(index) {
  const q = data.cases[index].quality;
  const rows = [
    ['Joint MAE', q.joint_mae_rad, data.thresholds.joint_mae_rad, 'rad'],
    ['Joint max absolute', q.joint_max_abs_rad, data.thresholds.joint_max_abs_rad, 'rad'],
    ['Gripper max absolute', q.gripper_max_abs, data.thresholds.gripper_max_abs, ''],
  ];
  $('quality-bars').innerHTML = rows.map(([name, value, threshold, unit]) => `<div class="gate-item"><div class="gate-info"><span>${name}</span><b>${fmt(value, 5)} ${unit}</b></div><div class="gate-track"><div class="gate-fill" style="width:${value / threshold * 100}%"></div></div><div class="gate-foot"><span>PASS · 阈值占用 ${fmt(value / threshold * 100, 1)}%</span><span>≤ ${threshold} ${unit}</span></div></div>`).join('');
}

function selectCase(index) {
  const c = data.cases[index];
  document.querySelectorAll('[data-case]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.case) === index)));
  document.querySelectorAll('[data-case-number]').forEach(n => n.textContent = '0' + (index + 1));
  $('case-task').textContent = caseNames[index];
  $('selected-speed').textContent = fmt(c.total_speedup);
  $('selected-original').textContent = fmt(c.original_ms) + ' ms';
  $('selected-final').textContent = fmt(c.final_ms) + ' ms';
  $('selected-saved').textContent = fmt(c.original_ms - c.final_ms) + ' ms';
  $('selected-reduction').textContent = fmt(c.latency_reduction_pct, 2) + '%';
  $('abba-before').textContent = fmt(c.sage_baseline_ms);
  $('abba-after').textContent = fmt(c.final_ms);
  $('abba-drop').textContent = fmt(c.lossless_reduction_pct) + '%';
  $('abba-speed').textContent = fmt(c.lossless_speedup, 5) + '×';
  $('recorded-change').textContent = (c.recorded_mae_degradation_rad < 0 ? '−' : '+') + fmt(Math.abs(c.recorded_mae_degradation_rad), 5) + ' rad';
  drawStages(index); drawAbba(index); drawQuality(index);
}

Promise.all(['summary', 'abba'].map(name => fetch(`data/${name}.json`).then(r => {
  if (!r.ok) throw new Error(`Data request failed: ${r.status}`);
  return r.json();
}))).then(([summary, measurements]) => {
  data = summary; abba = measurements;
  document.querySelectorAll('[data-case]').forEach(b => b.addEventListener('click', () => selectCase(Number(b.dataset.case))));
  selectCase(0);
}).catch(() => { $('data-error').hidden = false; });
