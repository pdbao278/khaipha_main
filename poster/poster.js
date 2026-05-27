// ============================================
// POSTER CHARTS - Chart.js Configuration
// Dự đoán nguy cơ Đột quỵ
// ============================================

// Toggle instructions panel
function toggleInstructions() {
    const panel = document.getElementById('instructionsPanel');
    const btn = document.getElementById('btnToggle');
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        btn.textContent = '📖 Ẩn hướng dẫn ▲';
    } else {
        panel.style.display = 'none';
        btn.textContent = '📖 Hướng dẫn ▼';
    }
}

// Download chart as PNG image
function downloadChart(canvasId, filename) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    // Create temporary download link
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

document.addEventListener('DOMContentLoaded', () => {
    createAlgorithmChart();
    createMIChart();
    setupTableChartSync();
});

// ============================================
// REAL-TIME SYNC: TABLE -> CHART
// ============================================
function setupTableChartSync() {
    const resultsTable = document.querySelector('.results-table');
    if (!resultsTable) return;

    const tbody = resultsTable.querySelector('tbody');
    if (!tbody) return;

    const syncChart = () => {
        if (!window.algoChartInstance) return;

        const rows = tbody.querySelectorAll('tr');
        const recalls = [];
        const precisions = [];
        const f1s = [];
        const aucs = [];
        const labels = [];

        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 5) {
                // Get algorithm label
                const label = cells[0].textContent.trim().replace(/\s+/g, '\n');
                labels.push(label);

                // Parse metrics
                const recall = parseFloat(cells[1].textContent.replace('%', '')) || 0;
                const precision = parseFloat(cells[2].textContent.replace('%', '')) || 0;
                const f1 = parseFloat(cells[3].textContent.replace('%', '')) || 0;
                const auc = parseFloat(cells[4].textContent.replace('%', '')) || 0;

                recalls.push(recall);
                precisions.push(precision);
                f1s.push(f1);
                aucs.push(auc);
            }
        });

        // Update Chart data and labels
        window.algoChartInstance.data.labels = labels;
        window.algoChartInstance.data.datasets[0].data = recalls;
        window.algoChartInstance.data.datasets[1].data = precisions;
        window.algoChartInstance.data.datasets[2].data = f1s;
        window.algoChartInstance.data.datasets[3].data = aucs;
        
        // Redraw
        window.algoChartInstance.update();
    };

    // Listen to multiple events for maximum reliability
    tbody.addEventListener('input', syncChart);
    tbody.addEventListener('keyup', syncChart);
    tbody.addEventListener('blur', syncChart, true);
}

// ============================================
// 1. BIỂU ĐỒ SO SÁNH 5 THUẬT TOÁN
// ============================================
function createAlgorithmChart() {
    const ctx = document.getElementById('algorithmChart');
    if (!ctx) return;

    const labels = [
        'Logistic\nRegression',
        'SVM\n(RBF)',
        'Random\nForest',
        'XGBoost',
        'Decision\nTree'
    ];

    window.algoChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Recall (%)',
                    data: [81.04, 85.57, 84.06, 83.05, 82.38],
                    backgroundColor: 'rgba(0, 121, 107, 0.8)',
                    borderColor: '#00796b',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.75,
                    categoryPercentage: 0.8,
                },
                {
                    label: 'Precision (%)',
                    data: [54.15, 49.85, 50.86, 50.67, 50.62],
                    backgroundColor: 'rgba(231, 76, 60, 0.75)',
                    borderColor: '#e74c3c',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.75,
                    categoryPercentage: 0.8,
                },
                {
                    label: 'F1-Score (%)',
                    data: [64.92, 63.00, 63.38, 62.94, 61.71],
                    backgroundColor: 'rgba(243, 156, 18, 0.8)',
                    borderColor: '#f39c12',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.75,
                    categoryPercentage: 0.8,
                },
                {
                    label: 'AUC-ROC (%)',
                    data: [85.07, 84.64, 84.50, 83.55, 83.51],
                    backgroundColor: 'rgba(142, 68, 173, 0.8)',
                    borderColor: '#8e44ad',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.75,
                    categoryPercentage: 0.8,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false, // disable for print
            layout: {
                padding: {
                    bottom: 12
                }
            },
            plugins: {
                title: {
                    display: false,
                    text: 'So sánh Hiệu năng 5 Thuật toán trên Test Set (Full 11 Features)',
                    font: { size: 13, weight: 'bold', family: 'Inter' },
                    color: '#1a237e',
                    padding: { bottom: 10 }
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { size: 11, family: 'Inter' },
                        usePointStyle: true,
                        pointStyle: 'rectRounded',
                        padding: 12
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        font: { size: 10, weight: '600', family: 'Inter' },
                        color: '#333',
                        maxRotation: 0
                    }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: {
                        color: 'rgba(0,0,0,0.06)',
                        drawBorder: false
                    },
                    ticks: {
                        font: { size: 10, family: 'Inter' },
                        callback: (v) => v + '%',
                        stepSize: 20
                    },
                    title: {
                        display: false,
                        text: 'Phần trăm (%)',
                        font: { size: 10, family: 'Inter', weight: '500' }
                    }
                }
            }
        },
        plugins: [{
            id: 'barLabels',
            afterDatasetsDraw(chart) {
                const { ctx } = chart;
                ctx.save();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.font = '500 7px Inter';
                ctx.fillStyle = '#555555';
                
                chart.data.datasets.forEach((dataset, datasetIndex) => {
                    const meta = chart.getDatasetMeta(datasetIndex);
                    if (meta.hidden) return;
                    
                    meta.data.forEach((bar, index) => {
                        const val = dataset.data[index];
                        if (val === null || val === undefined || isNaN(val) || val === 0) return;
                        
                        const x = bar.x;
                        const y = bar.y - 3;
                        ctx.fillText(val.toFixed(2) + '%', x, y);
                    });
                });
                ctx.restore();
            }
        }]
    });
}

// ============================================
// 2. BIỂU ĐỒ MUTUAL INFORMATION
// ============================================
function createMIChart() {
    const ctx = document.getElementById('miChart');
    if (!ctx) return;

    const features = [
        'Gender', 'SES', 'Smoking_Former', 'Smoking_Never',
        'Smoking_Current', 'BMI', 'Avg_Glucose',
        'Diabetes', 'Heart_Disease', 'Age', 'Hypertension'
    ];

    const scores = [
        0.000012, 0.000366, 0.000620, 0.000701,
        0.003104, 0.003639, 0.025014,
        0.028708, 0.036160, 0.068204, 0.085932
    ];

    const threshold = 0.01;
    const colors = scores.map(s => s >= threshold
        ? 'rgba(46, 204, 113, 0.85)'
        : 'rgba(231, 76, 60, 0.6)'
    );

    const borderColors = scores.map(s => s >= threshold
        ? '#27ae60'
        : '#c0392b'
    );

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'MI Score',
                data: scores,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 3,
                barPercentage: 0.7,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                title: {
                    display: false,
                    text: 'Feature Selection: Mutual Information vs Stroke Target',
                    font: { size: 12, weight: 'bold', family: 'Inter' },
                    color: '#1a237e',
                    padding: { bottom: 6 }
                },
                legend: { display: false },
                annotation: {
                    annotations: {
                        threshold: {
                            type: 'line',
                            xMin: threshold,
                            xMax: threshold,
                            borderColor: 'red',
                            borderWidth: 2,
                            borderDash: [5, 3],
                            label: {
                                display: true,
                                content: `Threshold = ${threshold}`,
                                position: 'end',
                                font: { size: 9 }
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(0,0,0,0.06)' },
                    ticks: {
                        font: { size: 9, family: 'Inter' },
                        callback: (v) => v.toFixed(2)
                    },
                    title: {
                        display: true,
                        text: 'Mutual Information Score',
                        font: { size: 10, family: 'Inter', weight: '500' }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        font: { size: 9, weight: '500', family: 'Inter' },
                        color: '#333'
                    }
                }
            }
        },
        plugins: [{
            // Custom plugin to draw threshold line
            id: 'thresholdLine',
            afterDraw(chart) {
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                const ctx = chart.ctx;
                const xPixel = xScale.getPixelForValue(threshold);

                ctx.save();
                ctx.beginPath();
                ctx.setLineDash([6, 3]);
                ctx.strokeStyle = '#e74c3c';
                ctx.lineWidth = 2;
                ctx.moveTo(xPixel, yScale.top);
                ctx.lineTo(xPixel, yScale.bottom);
                ctx.stroke();

                // Label
                ctx.setLineDash([]);
                ctx.fillStyle = '#e74c3c';
                ctx.font = 'bold 9px Inter';
                ctx.textAlign = 'left';
                ctx.fillText(`Threshold = ${threshold}`, xPixel + 4, yScale.top + 10);
                ctx.restore();
            }
        }]
    });
}
