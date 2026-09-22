/**
 * app.js - Frontend interactive client logic for DecodeLabs Project 2
 */

let elbowChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadElbowAnalysis();
  triggerTrainAndEvaluate();
  triggerPrediction();
});

function initEventListeners() {
  // Model Hyperparameters
  const kInput = document.getElementById("kInput");
  const kValBadge = document.getElementById("kValBadge");
  kInput.addEventListener("input", (e) => {
    kValBadge.textContent = e.target.value;
  });

  const splitInput = document.getElementById("splitInput");
  const splitValBadge = document.getElementById("splitValBadge");
  splitInput.addEventListener("input", (e) => {
    splitValBadge.textContent = `${e.target.value}%`;
  });

  document.getElementById("trainBtn").addEventListener("click", () => {
    triggerTrainAndEvaluate();
  });

  // Feature Input Sliders
  const inputs = [
    { id: "inSepalLen", valId: "valSepalLen" },
    { id: "inSepalWid", valId: "valSepalWid" },
    { id: "inPetalLen", valId: "valPetalLen" },
    { id: "inPetalWid", valId: "valPetalWid" },
  ];

  inputs.forEach(({ id, valId }) => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(valId);
    el.addEventListener("input", (e) => {
      valEl.textContent = `${parseFloat(e.target.value).toFixed(1)} cm`;
      triggerPrediction();
    });
  });
}

function applyPreset(sl, sw, pl, pw) {
  document.getElementById("inSepalLen").value = sl;
  document.getElementById("valSepalLen").textContent = `${sl.toFixed(1)} cm`;

  document.getElementById("inSepalWid").value = sw;
  document.getElementById("valSepalWid").textContent = `${sw.toFixed(1)} cm`;

  document.getElementById("inPetalLen").value = pl;
  document.getElementById("valPetalLen").textContent = `${pl.toFixed(1)} cm`;

  document.getElementById("inPetalWid").value = pw;
  document.getElementById("valPetalWid").textContent = `${pw.toFixed(1)} cm`;

  triggerPrediction();
}

async function loadElbowAnalysis() {
  try {
    const res = await fetch("/api/elbow-analysis?max_k=20");
    const data = await res.json();
    renderElbowChart(data);
  } catch (err) {
    console.error("Failed to load elbow data:", err);
  }
}

function renderElbowChart(data) {
  const ctx = document.getElementById("elbowChart").getContext("2d");

  if (elbowChartInstance) {
    elbowChartInstance.destroy();
  }

  elbowChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.k_range,
      datasets: [
        {
          label: "Test Error Rate",
          data: data.test_errors,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 4,
          pointBackgroundColor: "#3b82f6",
        },
        {
          label: "Train Error Rate",
          data: data.train_errors,
          borderColor: "#64748b",
          borderDash: [5, 5],
          borderWidth: 1.5,
          tension: 0.2,
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { family: "Inter", size: 11 } },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "K-Value (Neighbors)", color: "#64748b" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
        y: {
          title: { display: true, text: "Error Rate", color: "#64748b" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

async function triggerTrainAndEvaluate() {
  const k = parseInt(document.getElementById("kInput").value);
  const splitRatio = parseInt(document.getElementById("splitInput").value) / 100.0;

  try {
    const res = await fetch("/api/train-evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ k_neighbors: k, test_size: splitRatio }),
    });
    const data = await res.json();
    updateMetricsUI(data);
    triggerPrediction();
  } catch (err) {
    console.error("Training error:", err);
  }
}

function updateMetricsUI(data) {
  const m = data.metrics;
  document.getElementById("kpiAccuracy").textContent = `${(m.accuracy * 100).toFixed(1)}%`;
  document.getElementById("kpiF1").textContent = `${(m.weighted_f1_score * 100).toFixed(1)}%`;
  document.getElementById("kpiPrecision").textContent = `${(m.weighted_precision * 100).toFixed(1)}%`;
  document.getElementById("kpiRecall").textContent = `${(m.weighted_recall * 100).toFixed(1)}%`;

  // Render Confusion Matrix
  const cmGrid = document.getElementById("confusionGrid");
  cmGrid.innerHTML = "";
  const cm = m.confusion_matrix;
  const labels = ["Setosa", "Versicolor", "Virginica"];

  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const val = cm[r][c];
      const isCorrect = r === c;
      const cell = document.createElement("div");
      cell.className = `cm-cell ${isCorrect ? "cm-cell-correct" : ""}`;
      cell.innerHTML = `
        <div class="cm-val" style="color: ${isCorrect ? "#10b981" : val > 0 ? "#ef4444" : "#64748b"}">${val}</div>
        <div class="cm-label">T:${labels[r]} / P:${labels[c]}</div>
      `;
      cmGrid.appendChild(cell);
    }
  }

  // Render Benchmark Table
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = "";
  for (const [name, b] of Object.entries(data.benchmarks)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="color: #f8fafc; font-weight: 500;">${name}</td>
      <td style="color: #38bdf8;">${(b.accuracy * 100).toFixed(1)}%</td>
      <td style="color: #4ade80;">${(b.f1_score * 100).toFixed(1)}%</td>
      <td><span class="status-pill status-ready" style="font-size:0.7rem;">Verified</span></td>
    `;
    tbody.appendChild(tr);
  }
}

async function triggerPrediction() {
  const payload = {
    sepal_length: parseFloat(document.getElementById("inSepalLen").value),
    sepal_width: parseFloat(document.getElementById("inSepalWid").value),
    petal_length: parseFloat(document.getElementById("inPetalLen").value),
    petal_width: parseFloat(document.getElementById("inPetalWid").value),
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    renderPredictionResult(data);
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

function renderPredictionResult(data) {
  const species = data.predicted_species;
  const nameEl = document.getElementById("predSpeciesName");
  nameEl.textContent = `Iris ${species.charAt(0).toUpperCase() + species.slice(1)}`;
  
  if (species === "setosa") nameEl.style.color = "#38bdf8";
  else if (species === "versicolor") nameEl.style.color = "#4ade80";
  else nameEl.style.color = "#c084fc";

  document.getElementById("predConfidence").textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  const probs = data.class_probabilities;
  const pSetosa = Math.round(probs.setosa * 100);
  const pVersicolor = Math.round(probs.versicolor * 100);
  const pVirginica = Math.round(probs.virginica * 100);

  document.getElementById("probSetosa").style.width = `${pSetosa}%`;
  document.getElementById("probSetosaVal").textContent = `${pSetosa}%`;

  document.getElementById("probVersicolor").style.width = `${pVersicolor}%`;
  document.getElementById("probVersicolorVal").textContent = `${pVersicolor}%`;

  document.getElementById("probVirginica").style.width = `${pVirginica}%`;
  document.getElementById("probVirginicaVal").textContent = `${pVirginica}%`;
}
