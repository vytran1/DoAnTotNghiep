document.addEventListener("DOMContentLoaded", function () {
  const checkBtn = document.getElementById("check-btn");
  const statusDiv = document.getElementById("status");
  const resultDiv = document.getElementById("result-display");
  const predictionH2 = document.getElementById("prediction");
  const probabilityP = document.getElementById("probability");
  const chartCanvas = document.getElementById("shap-chart");

  let shapChart; // Biến để lưu trữ biểu đồ, cho phép hủy đi vẽ lại

  checkBtn.addEventListener("click", function () {
    // 1. Cập nhật giao diện
    statusDiv.textContent = "Đang lấy text từ trang...";
    resultDiv.style.display = "none";
    checkBtn.disabled = true;

    // 2. Gửi tin nhắn đến content.js để lấy text
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.sendMessage(
        tabs[0].id,
        { message: "get_page_text" },
        function (response) {
          if (response && response.text) {
            statusDiv.textContent = "Đang gọi AI (có thể mất vài giây)...";
            // 3. Đã có text, gọi API
            callBackendAPI(response.text);
          } else {
            statusDiv.textContent = "Lỗi: Không thể lấy text từ trang này.";
            checkBtn.disabled = false;
          }
        }
      );
    });
  });

  async function callBackendAPI(text) {
    try {
      const response = await fetch(
        "https://unhued-unaging-jayson.ngrok-free.dev/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: text }),
        }
      );

      if (!response.ok) {
        throw new Error(`Lỗi API: ${response.statusText}`);
      }

      const data = await response.json();

      // 4. Hiển thị kết quả
      displayResults(data);
    } catch (error) {
      statusDiv.textContent = `Lỗi: ${error.message}. Hãy chắc chắn API backend đang chạy.`;
    } finally {
      checkBtn.disabled = false;
      statusDiv.textContent = "";
    }
  }

  function displayResults(data) {
    resultDiv.style.display = "block";

    console.log(
      "BACKEND PREDICTION:",
      data.prediction,
      "PROB REAL:",
      data.probability_real
    );
    predictionH2.textContent = data.prediction;
    if (data.prediction === "REAL") {
      predictionH2.className = "real";
    } else {
      predictionH2.className = "fake";
    }

    const confidencePercent = data.probability_real * 100;

    probabilityP.textContent = `(Độ tin cậy: ${confidencePercent.toFixed(2)}%)`;

    // 5. Vẽ biểu đồ SHAP
    renderChart(data.top_features, data.base_value);
  }

  function renderChart(features, baseValue) {
    if (shapChart) {
      shapChart.destroy();
    }

    features.reverse();

    const labels = features.map((f) => f.feature);
    const values = features.map((f) => f.shap_value);

    const colors = values.map((v) =>
      v > 0 ? "rgba(75, 192, 192, 0.8)" : "rgba(255, 99, 132, 0.8)"
    );

    const ctx = chartCanvas.getContext("2d");
    shapChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Ảnh hưởng SHAP",
            data: values,
            backgroundColor: colors,
            borderColor: colors,
            borderWidth: 1,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: {
          legend: {
            display: false,
          },
          title: {
            display: true,
            text: `Các từ ảnh hưởng nhất (Base: ${baseValue.toFixed(4)})`,
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "SHAP Value",
            },
          },
        },
      },
    });
  }
});
