"use strict";

const chartColors = {
    primary: "#2563eb",
    primarySoft: "rgba(37, 99, 235, 0.14)",
    green: "#16a34a",
    amber: "#d97706",
    red: "#dc2626",
    purple: "#7c3aed",
    slate: "#64748b",
    grid: "rgba(100, 116, 139, 0.18)",
};

function readAnalyticsData() {
    const dataElement = document.getElementById("analyticsData");

    if (!dataElement) {
        return null;
    }

    try {
        return JSON.parse(dataElement.textContent);
    } catch (error) {
        console.error("Unable to parse analytics chart data.", error);
        return null;
    }
}

function createSohCycleChart(analytics) {
    const chartElement = document.getElementById("sohCycleChart");

    if (!chartElement || !analytics) {
        return;
    }

    new Chart(chartElement, {
        type: "line",
        data: {
            labels: analytics.cycle_labels || [],
            datasets: [
                {
                    label: "State of Health (%)",
                    data: analytics.soh_values || [],
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primarySoft,
                    borderWidth: 3,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: chartColors.primary,
                    pointBorderColor: "#ffffff",
                    pointBorderWidth: 2,
                    tension: 0.35,
                    fill: true,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: chartColors.slate,
                        usePointStyle: true,
                    },
                },
                tooltip: {
                    mode: "index",
                    intersect: false,
                },
            },
            interaction: {
                mode: "nearest",
                axis: "x",
                intersect: false,
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Cycle",
                        color: chartColors.slate,
                    },
                    grid: {
                        color: chartColors.grid,
                    },
                    ticks: {
                        color: chartColors.slate,
                    },
                },
                y: {
                    title: {
                        display: true,
                        text: "SOH (%)",
                        color: chartColors.slate,
                    },
                    beginAtZero: false,
                    grid: {
                        color: chartColors.grid,
                    },
                    ticks: {
                        color: chartColors.slate,
                    },
                },
            },
        },
    });
}

function createConditionPieChart(analytics) {
    const chartElement = document.getElementById("conditionPieChart");

    if (!chartElement || !analytics) {
        return;
    }

    new Chart(chartElement, {
        type: "pie",
        data: {
            labels: analytics.condition_labels || [],
            datasets: [
                {
                    label: "Battery Conditions",
                    data: analytics.condition_values || [],
                    backgroundColor: [
                        chartColors.green,
                        chartColors.primary,
                        chartColors.amber,
                        chartColors.red,
                        chartColors.purple,
                    ],
                    borderColor: "#ffffff",
                    borderWidth: 3,
                    hoverOffset: 8,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: chartColors.slate,
                        padding: 18,
                        usePointStyle: true,
                    },
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const label = context.label || "Condition";
                            const value = context.parsed || 0;
                            return `${label}: ${value}`;
                        },
                    },
                },
            },
        },
    });
}

function initializeAnalyticsCharts() {
    if (typeof Chart === "undefined") {
        console.error("Chart.js is required before charts.js.");
        return;
    }

    const analytics = readAnalyticsData();
    createSohCycleChart(analytics);
    createConditionPieChart(analytics);
}

document.addEventListener("DOMContentLoaded", initializeAnalyticsCharts);
// -----------------------------
// History Page Search
// -----------------------------
document.addEventListener("DOMContentLoaded", function () {

    const searchBox = document.getElementById("historySearch");
    const table = document.getElementById("historyTable");

    if (!searchBox || !table) return;

    searchBox.addEventListener("keyup", function () {

        const filter = this.value.toLowerCase();
        const rows = table.getElementsByTagName("tbody")[0].getElementsByTagName("tr");

        for (let i = 0; i < rows.length; i++) {

            const text = rows[i].innerText.toLowerCase();

            if (text.indexOf(filter) > -1) {
                rows[i].style.display = "";
            } else {
                rows[i].style.display = "none";
            }

        }

    });

});
